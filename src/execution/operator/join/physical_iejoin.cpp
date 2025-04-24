#include "duckdb/execution/operator/join/physical_iejoin.hpp"

#include "duckdb/common/atomic.hpp"
#include "duckdb/common/operator/comparison_operators.hpp"
#include "duckdb/common/row_operations/row_operations.hpp"
#include "duckdb/common/sort/sort.hpp"
#include "duckdb/common/sort/sorted_block.hpp"
#include "duckdb/common/thread.hpp"
#include "duckdb/common/vector_operations/vector_operations.hpp"
#include "duckdb/execution/expression_executor.hpp"
#include "duckdb/main/client_context.hpp"
#include "duckdb/parallel/event.hpp"
#include "duckdb/parallel/meta_pipeline.hpp"
#include "duckdb/parallel/thread_context.hpp"
#include "duckdb/planner/expression/bound_constant_expression.hpp"
#include "duckdb/planner/expression/bound_reference_expression.hpp"

namespace duckdb {

PhysicalIEJoin::PhysicalIEJoin(LogicalComparisonJoin &op, PhysicalOperator &left, PhysicalOperator &right,
                               vector<JoinCondition> cond, JoinType join_type, idx_t estimated_cardinality)
    : PhysicalRangeJoin(op, PhysicalOperatorType::IE_JOIN, left, right, std::move(cond), join_type,
                        estimated_cardinality) {

	// 1. let L1 (resp. L2) be the array of column X (resp. Y)
	D_ASSERT(conditions.size() >= 2);
	for (idx_t i = 0; i < 2; ++i) {
		auto &cond = conditions[i];
		D_ASSERT(cond.left->return_type == cond.right->return_type);
		join_key_types.push_back(cond.left->return_type);

		// Convert the conditions to sort orders
		auto left = cond.left->Copy();
		auto right = cond.right->Copy();
		auto sense = OrderType::INVALID;

		// 2. if (op1 ∈ {>, ≥}) sort L1 in descending order
		// 3. else if (op1 ∈ {<, ≤}) sort L1 in ascending order
		// 4. if (op2 ∈ {>, ≥}) sort L2 in ascending order
		// 5. else if (op2 ∈ {<, ≤}) sort L2 in descending order
		switch (cond.comparison) {
		case ExpressionType::COMPARE_GREATERTHAN:
		case ExpressionType::COMPARE_GREATERTHANOREQUALTO:
			sense = i ? OrderType::ASCENDING : OrderType::DESCENDING;
			break;
		case ExpressionType::COMPARE_LESSTHAN:
		case ExpressionType::COMPARE_LESSTHANOREQUALTO:
			sense = i ? OrderType::DESCENDING : OrderType::ASCENDING;
			break;
		default:
			throw NotImplementedException("Unimplemented join type for IEJoin");
		}
		lhs_orders.emplace_back(sense, OrderByNullType::NULLS_LAST, std::move(left));
		rhs_orders.emplace_back(sense, OrderByNullType::NULLS_LAST, std::move(right));
	}

	// Debugging output to verify the join key expressions being used for sorting
	// These expressions determine the sort order for each input side (LHS and RHS)
	// and are critical for the correct operation of the IEJoin algorithm.
	// 
	// LHS0 and RHS0 correspond to the first join predicate (e.g., start time)
	// LHS1 and RHS1 correspond to the second join predicate (e.g., end time)
	// 
	// This helps verify that the sort directions (ascending/descending) are 
	// derived correctly from the comparison types (>, <, etc.)
	fprintf(stderr, "LHS0: %s\n", lhs_orders[0].expression->ToString().c_str());
	fprintf(stderr, "RHS0: %s\n", rhs_orders[0].expression->ToString().c_str());
	fprintf(stderr, "LHS1: %s\n", lhs_orders[1].expression->ToString().c_str());
	fprintf(stderr, "RHS1: %s\n", rhs_orders[1].expression->ToString().c_str());

	for (idx_t i = 2; i < conditions.size(); ++i) {
		auto &cond = conditions[i];
		D_ASSERT(cond.left->return_type == cond.right->return_type);
		join_key_types.push_back(cond.left->return_type);
	}
}

//===--------------------------------------------------------------------===//
// Sink
//===--------------------------------------------------------------------===//
class IEJoinLocalState : public LocalSinkState {
public:
	using LocalSortedTable = PhysicalRangeJoin::LocalSortedTable;

	IEJoinLocalState(ClientContext &context, const PhysicalRangeJoin &op, const idx_t child)
	    : table(context, op, child) {
	}

	//! The local sort state
	LocalSortedTable table;
};

class IEJoinGlobalState : public GlobalSinkState {
public:
	using GlobalSortedTable = PhysicalRangeJoin::GlobalSortedTable;

public:
	IEJoinGlobalState(ClientContext &context, const PhysicalIEJoin &op) : child(0) {
		tables.resize(2);
		RowLayout lhs_layout;
		lhs_layout.Initialize(op.children[0].get().GetTypes());
		vector<BoundOrderByNode> lhs_order;
		lhs_order.emplace_back(op.lhs_orders[0].Copy());
		tables[0] = make_uniq<GlobalSortedTable>(context, lhs_order, lhs_layout, op);

		RowLayout rhs_layout;
		rhs_layout.Initialize(op.children[1].get().GetTypes());
		vector<BoundOrderByNode> rhs_order;
		rhs_order.emplace_back(op.rhs_orders[0].Copy());
		tables[1] = make_uniq<GlobalSortedTable>(context, rhs_order, rhs_layout, op);
	}

	IEJoinGlobalState(IEJoinGlobalState &prev) : tables(std::move(prev.tables)), child(prev.child + 1) {
		state = prev.state;
	}

	void Sink(DataChunk &input, IEJoinLocalState &lstate) {
		auto &table = *tables[child];
		auto &global_sort_state = table.global_sort_state;
		auto &local_sort_state = lstate.table.local_sort_state;

		// Sink the data into the local sort state
		lstate.table.Sink(input, global_sort_state);

		// When sorting data reaches a certain size, we sort it
		if (local_sort_state.SizeInBytes() >= table.memory_per_thread) {
			local_sort_state.Sort(global_sort_state, true);
		}
	}

	vector<unique_ptr<GlobalSortedTable>> tables;
	size_t child;
};

unique_ptr<GlobalSinkState> PhysicalIEJoin::GetGlobalSinkState(ClientContext &context) const {
	D_ASSERT(!sink_state);
	return make_uniq<IEJoinGlobalState>(context, *this);
}

unique_ptr<LocalSinkState> PhysicalIEJoin::GetLocalSinkState(ExecutionContext &context) const {
	idx_t sink_child = 0;
	if (sink_state) {
		const auto &ie_sink = sink_state->Cast<IEJoinGlobalState>();
		sink_child = ie_sink.child;
	}
	return make_uniq<IEJoinLocalState>(context.client, *this, sink_child);
}

SinkResultType PhysicalIEJoin::Sink(ExecutionContext &context, DataChunk &chunk, OperatorSinkInput &input) const {
	auto &gstate = input.global_state.Cast<IEJoinGlobalState>();
	auto &lstate = input.local_state.Cast<IEJoinLocalState>();

	gstate.Sink(chunk, lstate);

	return SinkResultType::NEED_MORE_INPUT;
}

SinkCombineResultType PhysicalIEJoin::Combine(ExecutionContext &context, OperatorSinkCombineInput &input) const {
	auto &gstate = input.global_state.Cast<IEJoinGlobalState>();
	auto &lstate = input.local_state.Cast<IEJoinLocalState>();
	gstate.tables[gstate.child]->Combine(lstate.table);
	auto &client_profiler = QueryProfiler::Get(context.client);

	context.thread.profiler.Flush(*this);
	client_profiler.Flush(context.thread.profiler);

	return SinkCombineResultType::FINISHED;
}

//===--------------------------------------------------------------------===//
// Finalize
//===--------------------------------------------------------------------===//
SinkFinalizeType PhysicalIEJoin::Finalize(Pipeline &pipeline, Event &event, ClientContext &context,
                                          OperatorSinkFinalizeInput &input) const {
	auto &gstate = input.global_state.Cast<IEJoinGlobalState>();
	auto &table = *gstate.tables[gstate.child];
	auto &global_sort_state = table.global_sort_state;

	if ((gstate.child == 1 && PropagatesBuildSide(join_type)) || (gstate.child == 0 && IsLeftOuterJoin(join_type))) {
		// for FULL/LEFT/RIGHT OUTER JOIN, initialize found_match to false for every tuple
		table.IntializeMatches();
	}
	if (gstate.child == 1 && global_sort_state.sorted_blocks.empty() && EmptyResultIfRHSIsEmpty()) {
		// Empty input!
		return SinkFinalizeType::NO_OUTPUT_POSSIBLE;
	}

	// Sort the current input child
	table.Finalize(pipeline, event);

	// Move to the next input child
	++gstate.child;

	return SinkFinalizeType::READY;
}

//===--------------------------------------------------------------------===//
// Operator
//===--------------------------------------------------------------------===//
OperatorResultType PhysicalIEJoin::ExecuteInternal(ExecutionContext &context, DataChunk &input, DataChunk &chunk,
                                                   GlobalOperatorState &gstate, OperatorState &state) const {
	return OperatorResultType::FINISHED;
}

//===--------------------------------------------------------------------===//
// Source
//===--------------------------------------------------------------------===//
struct IEJoinUnion {
	using SortedTable = PhysicalRangeJoin::GlobalSortedTable;

	static idx_t AppendKey(SortedTable &table, ExpressionExecutor &executor, SortedTable &marked, int64_t increment,
	                       int64_t base, const idx_t block_idx);

	static void Sort(SortedTable &table) {
		auto &global_sort_state = table.global_sort_state;
		global_sort_state.PrepareMergePhase();
		while (global_sort_state.sorted_blocks.size() > 1) {
			global_sort_state.InitializeMergeRound();
			MergeSorter merge_sorter(global_sort_state, global_sort_state.buffer_manager);
			merge_sorter.PerformInMergeRound();
			global_sort_state.CompleteMergeRound(true);
		}
	}

	template <typename T>
	static vector<T> ExtractColumn(SortedTable &table, idx_t col_idx) {
		vector<T> result;
		result.reserve(table.count);

		auto &gstate = table.global_sort_state;
		auto &blocks = *gstate.sorted_blocks[0]->payload_data;
		PayloadScanner scanner(blocks, gstate, false);

		DataChunk payload;
		payload.Initialize(Allocator::DefaultAllocator(), gstate.payload_layout.GetTypes());
		for (;;) {
			payload.Reset();
			scanner.Scan(payload);
			const auto count = payload.size();
			if (!count) {
				break;
			}

			const auto data_ptr = FlatVector::GetData<T>(payload.data[col_idx]);
			result.insert(result.end(), data_ptr, data_ptr + count);
		}

		return result;
	}

	IEJoinUnion(ClientContext &context, const PhysicalIEJoin &op, SortedTable &t1, const idx_t b1, SortedTable &t2,
	            const idx_t b2);

	idx_t SearchL1(idx_t pos);
	bool NextRow();

	//! Inverted loop
	idx_t JoinComplexBlocks(SelectionVector &lsel, SelectionVector &rsel);

	//! L1
	unique_ptr<SortedTable> l1;
	//! L2
	unique_ptr<SortedTable> l2;

	//! Li
	vector<int64_t> li;
	//! P
	vector<idx_t> p;

	//! B
	vector<validity_t> bit_array;
	ValidityMask bit_mask;

	//! Bloom Filter
	static constexpr idx_t BLOOM_CHUNK_BITS = 1024;
	idx_t bloom_count;
	vector<validity_t> bloom_array;
	ValidityMask bloom_filter;

	//! Iteration state
	idx_t n;
	idx_t i;
	idx_t j;
	unique_ptr<SBIterator> op1;
	unique_ptr<SBIterator> off1;
	unique_ptr<SBIterator> op2;
	unique_ptr<SBIterator> off2;
	int64_t lrid;
	

	//===--------------------------------------------------------------------===//
	// Sweepline Join Structures & Methods
	//===--------------------------------------------------------------------===//

	// Represents a timestamped event used in the sweep line algorithm for overlap joins
	// Each row contributes two events: one for interval start and one for interval end
	// These are sorted chronologically to simulate a line sweeping through time
	struct SweeplineEvent {
		timestamp_t time;// The timestamp of the event (start or end of an interval)
		bool is_start; // True if this is a start event, false for end event
		int64_t rid; // going to borrow from IEJoinUnion
    
    	// For sorting events
		bool operator<(const SweeplineEvent &other) const {
			if (time.value != other.time.value) 
				return time.value < other.time.value;
			// For same time, end events before start events
				// !!!IMPORTANT!!!
					// had issue with equivalent conditions
			return is_start > other.is_start;
		}
	};

	// Payload columns extracted from the sorted IEJoin table
	// - l1_rid: Row IDs for all intervals (positive = LHS, negative = RHS)
	// - cmp1_col, cmp2_col: Start and end timestamps, reused depending on side
	vector<timestamp_t> l1_rid;
	vector<timestamp_t> cmp1_col; // Usually end time
	vector<timestamp_t> cmp2_col; // Usually start time

	// Core event list for the sweepline
	vector<SweeplineEvent> events;
	idx_t current_event; // Pointer into `events` for iteration

	// Currently "active" intervals during sweep
	// These represent rows whose intervals are open and may match incoming events
	std::set<idx_t> active_lhs;
	std::set<idx_t> active_rhs;
	// check duplicates - for some reason this also removes a small amount of rows
	// std::set<std::pair<int64_t, int64_t>> unique_results;

	// Initialize the sweepline event list based on extracted intervals
	void InitializeSweepline();
	// Process a single event: update active sets and optionally emit join matches
	bool ProcessNextEvent();
	// Executes the full overlap join using the sweepline algorithm
	// Fills `lsel` and `rsel` with join pair indices
	idx_t SweeplineJoinBlocks(SelectionVector &lsel, SelectionVector &rsel);
};

idx_t IEJoinUnion::AppendKey(SortedTable &table, ExpressionExecutor &executor, SortedTable &marked, int64_t increment,
                             int64_t base, const idx_t block_idx) {
	LocalSortState local_sort_state;
	local_sort_state.Initialize(marked.global_sort_state, marked.global_sort_state.buffer_manager);

	// Reading
	const auto valid = table.count - table.has_null;
	auto &gstate = table.global_sort_state;
	PayloadScanner scanner(gstate, block_idx);
	auto table_idx = block_idx * gstate.block_capacity;

	DataChunk scanned;
	scanned.Initialize(Allocator::DefaultAllocator(), scanner.GetPayloadTypes());

	// Writing
	auto types = local_sort_state.sort_layout->logical_types;
	const idx_t payload_idx = types.size();

	const auto &payload_types = local_sort_state.payload_layout->GetTypes();
	types.insert(types.end(), payload_types.begin(), payload_types.end());
	const idx_t rid_idx = types.size() - 1;

	DataChunk keys;
	DataChunk payload;
	keys.Initialize(Allocator::DefaultAllocator(), types);

	idx_t inserted = 0;
	for (auto rid = base; table_idx < valid;) {
		scanned.Reset();
		scanner.Scan(scanned);

		// NULLs are at the end, so stop when we reach them
		auto scan_count = scanned.size();
		if (table_idx + scan_count > valid) {
			scan_count = valid - table_idx;
			scanned.SetCardinality(scan_count);
		}
		if (scan_count == 0) {
			break;
		}
		table_idx += scan_count;

		// Compute the input columns from the payload
		keys.Reset();
		keys.Split(payload, rid_idx);
		//fprintf(stderr, "expressions.size() = %zu, columns = %llu\n", scanned.size(), keys.ColumnCount());
		executor.Execute(scanned, keys);

		// Mark the rid column
		payload.data[0].Sequence(rid, increment, scan_count);
		payload.SetCardinality(scan_count);
		keys.Fuse(payload);
		rid += increment * UnsafeNumericCast<int64_t>(scan_count);

		// Sort on the sort columns (which will no longer be needed)
		keys.Split(payload, payload_idx);
		local_sort_state.SinkChunk(keys, payload);
		inserted += scan_count;
		keys.Fuse(payload);

		// Flush when we have enough data
		if (local_sort_state.SizeInBytes() >= marked.memory_per_thread) {
			local_sort_state.Sort(marked.global_sort_state, true);
		}
	}
	marked.global_sort_state.AddLocalState(local_sort_state);
	marked.count += inserted;

	return inserted;
}

IEJoinUnion::IEJoinUnion(ClientContext &context, const PhysicalIEJoin &op, SortedTable &t1, const idx_t b1,
                         SortedTable &t2, const idx_t b2)
    : n(0), i(0) {
	// input : query Q with 2 join predicates t1.X op1 t2.X' and t1.Y op2 t2.Y', tables T, T' of sizes m and n resp.
	// output: a list of tuple pairs (ti , tj)
	// Note that T/T' are already sorted on X/X' and contain the payload data
	// We only join the two block numbers and use the sizes of the blocks as the counts

	// 0. Filter out tables with no overlap
	if (!t1.BlockSize(b1) || !t2.BlockSize(b2)) {
		return;
	}

	const auto &cmp1 = op.conditions[0].comparison;
	SBIterator bounds1(t1.global_sort_state, cmp1);
	SBIterator bounds2(t2.global_sort_state, cmp1);

	// t1.X[0] op1 t2.X'[-1]
	bounds1.SetIndex(bounds1.block_capacity * b1);
	bounds2.SetIndex(bounds2.block_capacity * b2 + t2.BlockSize(b2) - 1);
	if (!bounds1.Compare(bounds2)) {
		return;
	}

	// 1. let L1 (resp. L2) be the array of column X (resp. Y )
	const auto &order1 = op.lhs_orders[0];
	const auto &order2 = op.lhs_orders[1];
	// Extract the join order expressions for the RHS (right-hand side) input:
	// - order3 corresponds to the first join predicate (typically interval start)
	// - order4 corresponds to the second join predicate (typically interval end)
	// These determine how the RHS input will be sorted and joined using the IEJoin logic
	const auto &order3 = op.rhs_orders[0];
	const auto &order4 = op.rhs_orders[1];

	// 2. if (op1 ∈ {>, ≥}) sort L1 in descending order
	// 3. else if (op1 ∈ {<, ≤}) sort L1 in ascending order

	// For the union algorithm, we make a unified table with the keys and the rids as the payload:
	//		X/X', Y/Y', R/R'/Li
	// The first position is the sort key.
	vector<LogicalType> types;
	// added
	types.emplace_back(order1.expression->return_type);
	types.emplace_back(order2.expression->return_type);
	types.emplace_back(LogicalType::BIGINT);
	RowLayout payload_layout;
	payload_layout.Initialize(types);

	// Sort on the first expression
	auto ref = make_uniq<BoundReferenceExpression>(order1.expression->return_type, 0U);
	vector<BoundOrderByNode> orders;
	orders.emplace_back(order1.type, order1.null_order, std::move(ref));
	
	// goal comment
	auto from_left = make_uniq<BoundConstantExpression>(Value::BOOLEAN(true));
	orders.emplace_back(SBIterator::ComparisonValue(cmp1) == 0 ? OrderType::DESCENDING : OrderType::ASCENDING,
	                    OrderByNullType::ORDER_DEFAULT, std::move(from_left));

	l1 = make_uniq<SortedTable>(context, orders, payload_layout, op);

	// LHS has positive rids
	ExpressionExecutor l_executor(context);
	l_executor.AddExpression(*order1.expression);
	// add const column true
	auto left_const = make_uniq<BoundConstantExpression>(Value::BOOLEAN(true));
	l_executor.AddExpression(*left_const);
	l_executor.AddExpression(*order2.expression);
	// extract key columns from the input for sorting
	l_executor.AddExpression(*order1.expression);
	//Append LHS data into the unified sorted table `l1`	
	AppendKey(t1, l_executor, *l1, 1, 1, b1);
	// Debug message to confirm that LHS data append completed
	fprintf(stderr, "\n made it past first Append\n");

	// RHS has negative rids
	ExpressionExecutor r_executor(context);
	r_executor.AddExpression(*op.rhs_orders[0].expression);
	// add const column flase
	auto right_const = make_uniq<BoundConstantExpression>(Value::BOOLEAN(false));
	r_executor.AddExpression(*right_const);
	r_executor.AddExpression(*op.rhs_orders[1].expression);
	// Add the RHS sort expression (e.g., interval start or end) to the executor
	r_executor.AddExpression(*op.rhs_orders[0].expression);
	// Append RHS data into the unified sorted table `l1`
	AppendKey(t2, r_executor, *l1, -1, -1, b2);
	fprintf(stderr, "\n made it past second Append\n");
	// Bail if no valid data was appended to l1
	if (l1->global_sort_state.sorted_blocks.empty()) {
		return;
	}
	// Final sort on l1 using the primary key expression
	Sort(*l1);
	// Build index iterators on the primary join key
	op1 = make_uniq<SBIterator>(l1->global_sort_state, cmp1);
	off1 = make_uniq<SBIterator>(l1->global_sort_state, cmp1);

	// We don't actually need the L1 column, just its sort key, which is in the sort blocks
	li = ExtractColumn<int64_t>(*l1, types.size() - 1);

	fprintf(stderr, "\n made it past sorting l1? \n");

	l1_rid = ExtractColumn<timestamp_t>(*l1,2); // RID
	cmp1_col = ExtractColumn<timestamp_t>(*l1,1);// usually end
	cmp2_col = ExtractColumn<timestamp_t>(*l1, 0); // usually start

	fprintf(stderr, "\n check if we accessed 2nd index\n");

	fprintf(stderr, "Number of columns in l1: %zu\n", l1->global_sort_state.payload_layout.GetTypes().size());
	// Print sample data for debugging
	for (size_t i = 0; i < 5000; i+=50)
	{
		fprintf(stderr, "index l1[2] %llu: %lld\n", 
		static_cast<unsigned long long>(i), 
		static_cast<long long>(l1_rid[i].value));
		fprintf(stderr, "comparison 1 l1[1] %llu: %lld\n", 
			static_cast<unsigned long long>(i), 
			static_cast<long long>(cmp1_col[i].value));
		fprintf(stderr, "comparison 2 l1[0] %llu: %lld\n", 
		static_cast<unsigned long long>(i), 
		static_cast<long long>(cmp2_col[i].value));
	}

	// 4. if (op2 ∈ {>, ≥}) sort L2 in ascending order
	// 5. else if (op2 ∈ {<, ≤}) sort L2 in descending order

	// We sort on Y/Y' to obtain the sort keys and the permutation array.
	// For this we just need a two-column table of Y, P
	types.clear();
	types.emplace_back(LogicalType::BIGINT);
	payload_layout.Initialize(types);

	// Sort on the first expression
	orders.clear();
	ref = make_uniq<BoundReferenceExpression>(order2.expression->return_type, 0U);
	orders.emplace_back(order2.type, order2.null_order, std::move(ref));

	// Prepare executor and table for sorting L2
	ExpressionExecutor executor(context);
	executor.AddExpression(*orders[0].expression);

	// Append all blocks from l1 into l2 to compute the permutation
	l2 = make_uniq<SortedTable>(context, orders, payload_layout, op);
	for (idx_t base = 0, block_idx = 0; block_idx < l1->BlockCount(); ++block_idx) {
		base += AppendKey(*l1, executor, *l2, 1, NumericCast<int64_t>(base), block_idx);
	}

	Sort(*l2);

	// We don't actually need the L2 column, just its sort key, which is in the sort blocks

	// 6. compute the permutation array P of L2 w.r.t. L1
	p = ExtractColumn<idx_t>(*l2, types.size() - 1);



	n = l2->count.load(); // number of joined rows
	fprintf(stderr, "\n L2 count = %zu\n", n);
	/*
	Todo: fix materialization

	vector<timestamp_t> left_starts = ExtractColumn<timestamp_t>(*l2, 0);
	vector<timestamp_t> left_ends = ExtractColumn<timestamp_t>(*l2, 1);
	vector<timestamp_t> right_starts = ExtractColumn<timestamp_t>(*l1,1);
	vector<timestamp_t> right_ends = ExtractColumn<timestamp_t>(*l1, 0);
	fprintf(stderr, "\n Thread#  %zu - Column extraction successful: left_starts = %zu, right_starts = %zu, right_ends = %zu\n",
    std::hash<std::thread::id>{}(std::this_thread::get_id()),
    left_starts.size(),/* left_ends.size(),*/ 
	/*right_starts.size(), right_ends.size());
	*/

	InitializeSweepline();
	fprintf(stderr, "\nCreated %zu events in total\n", events.size());
	// sort the events
	// Pre-declare iterator to avoid repeated construction (if used)
	SBIterator sort_iter1(l1->global_sort_state, op.conditions[0].comparison);
    SBIterator sort_iter2(l1->global_sort_state, op.conditions[0].comparison);

	/*std::sort(events.begin(), events.end(),(const Event& a, const Event& b) {
	   // error for iterators
	
	   SBIterator iter1(l1->global_sort_state, op.conditions[0].comparison);
	   SBIterator iter2(l1->global_sort_state, op.conditions[0].comparison);
	   
	   //fprintf(stderr, "Sorting events - check out of bounds!!\n");

	   D_ASSERT(a.idx < l1->count.load());
	   D_ASSERT(b.idx < l1->count.load());
	   
	   // getting an out of bounds issue
	   iter1.SetIndex(std::min(a.idx, n-1));
	   iter2.SetIndex(std::min(b.idx, n-1));
	   
	   if (iter1.Compare(iter2) != iter2.Compare(iter1)) {
		   return iter1.Compare(iter2);
	   }
	   
	   // Then by event type 
	   if (a.type != b.type) {
		   return a.type < b.type;
	   }
	   
	   // Then by table side (RHS before LHS for START, LHS before RHS for END)
	   return (a.type == Event::START) ? !a.is_left : a.is_left;
	});*/

	/*fprintf(stderr, "\nSorted %zu events\n", events.size());

	for (size_t i = 0; i < std::min(events.size(), size_t(10)); i++) {
        fprintf(stderr, "Event %zu: type=%s, idx=%zu, is_left=%d\n", 
                i, 
                events[i].type == Event::START ? "START" : "END", 
                events[i].idx, 
                events[i].is_left);
    }*/

	// -- debugging
		// ---
	// --- 



	/*for (size_t i = 0; i < 3; i++)
	{
		fprintf(stderr, "%llu: %lld\n", 
		static_cast<unsigned long long>(i), 
		static_cast<long long>(right_starts[i].value));
		fprintf(stderr, "%llu: %lld\n", 
		static_cast<unsigned long long>(i), 
		static_cast<long long>(right_ends[i].value));
		fprintf(stderr, "%llu: %lld\n", 
			static_cast<unsigned long long>(i), 
			static_cast<long long>(left_starts[i].value));
	}*/

	

	// Print column headers for left_starts and right_starts
	fprintf(stderr, "l1[0] = %s\n", order1.expression->GetName().c_str());
	fprintf(stderr, "l1[2] = %s\n", order2.expression->GetName().c_str());
	//fprintf(stderr, "l2[0] = %s\n", order2.expression->GetName().c_str());

	(void) ProcessNextEvent();
	fprintf(stderr, "\n init process Next Event\n");
	

	// build a sweepline instead of bloom filter and bit array
	/*
	// 7. initialize bit-array B (|B| = n), and set all bits to 0
	n = l2->count.load();
	bit_array.resize(ValidityMask::EntryCount(n), 0);
	bit_mask.Initialize(bit_array.data(), n);

	// Bloom filter
	bloom_count = (n + (BLOOM_CHUNK_BITS - 1)) / BLOOM_CHUNK_BITS;
	bloom_array.resize(ValidityMask::EntryCount(bloom_count), 0);
	bloom_filter.Initialize(bloom_array.data(), bloom_count);

	// 11. for(i←1 to n) do
	const auto &cmp2 = op.conditions[1].comparison;
	op2 = make_uniq<SBIterator>(l2->global_sort_state, cmp2);
	off2 = make_uniq<SBIterator>(l2->global_sort_state, cmp2);
	i = 0;
	j = 0;
	(void)NextRow(); // call this to init it?
	*/
}

// Initializes the sweepline event structure for overlap join logic
// Converts start/end intervals into a flat list of timestamped events
// These events will later be processed in order to compute overlaps
void IEJoinUnion::InitializeSweepline() {
	// Extract interval columns if not already done
	if (l1_rid.empty()) {
        l1_rid = ExtractColumn<timestamp_t>(*l1, 2);
        cmp1_col = ExtractColumn<timestamp_t>(*l1, 1);
        cmp2_col = ExtractColumn<timestamp_t>(*l1, 0);
    }
    
    // Prepare the event list and reset state
    events.clear();
    events.reserve(l1_rid.size() * 2); // each row 2 events 
    active_lhs.clear();
    active_rhs.clear();
    current_event = 0;
    
    // init
    for (size_t i = 0; i < l1_rid.size(); i++) {
        int64_t row_id = l1_rid[i].value;
        bool is_lhs = row_id > 0;
		

		// LHS events: [start = cmp2_col, end = cmp1_col]
        // RHS events: [start = cmp1_col, end = cmp2_col]
        if (is_lhs) {
            // For LHS (b): cmp1_col = b.end, cmp2_col = b.start
            events.push_back({cmp2_col[i], true, row_id});   //start
            events.push_back({cmp1_col[i], false, row_id});  // end
        } else {
            // For RHS (a): cmp1_col = a.start, cmp2_col = a.end
            events.push_back({cmp1_col[i], true, row_id});
            events.push_back({cmp2_col[i], false, row_id});
        }
    }
    
    // Sort by time
    std::sort(events.begin(), events.end());
}

bool IEJoinUnion::ProcessNextEvent() {
	if (current_event >= events.size()) {
        return false; // No more events to process
    }
    
    // Get the current event
    const auto& event = events[current_event++];
    int64_t row_id = event.rid;
    bool is_lhs = row_id > 0;
    
    if (event.is_start) {
        // Start event - add to active set and generate results
        if (is_lhs) {
            active_lhs.insert(row_id);
        } else {
            active_rhs.insert(row_id);
        }
    } else {
        // End event - remove from active set
        if (is_lhs) {
            active_lhs.erase(row_id);
        } else {
            active_rhs.erase(row_id);
        }
    }
    
    return true;
}

bool IEJoinUnion::NextRow() {
	for (; i < n; ++i) {
		// 12. pos ← P[i]
		auto pos = p[i];
		lrid = li[pos];
		if (lrid < 0) {
			continue;
		}

		// 16. B[pos] ← 1
		op2->SetIndex(i);
		for (; off2->GetIndex() < n; ++(*off2)) {
			if (!off2->Compare(*op2)) {
				break;
			}
			const auto p2 = p[off2->GetIndex()];
			if (li[p2] < 0) {
				// Only mark rhs matches.
				bit_mask.SetValid(p2);
				bloom_filter.SetValid(p2 / BLOOM_CHUNK_BITS);
			}
		}

		// 9.  if (op1 ∈ {≤,≥} and op2 ∈ {≤,≥}) eqOff = 0
		// 10. else eqOff = 1
		// No, because there could be more than one equal value.
		// Find the leftmost off1 where L1[pos] op1 L1[off1..n]
		// These are the rows that satisfy the op1 condition
		// and that is where we should start scanning B from
		j = pos;

		return true;
	}
	return false;
}

static idx_t NextValid(const ValidityMask &bits, idx_t j, const idx_t n) {
	if (j >= n) {
		return n;
	}

	// We can do a first approximation by checking entries one at a time
	// which gives 64:1.
	idx_t entry_idx, idx_in_entry;
	bits.GetEntryIndex(j, entry_idx, idx_in_entry);
	auto entry = bits.GetValidityEntry(entry_idx++);

	// Trim the bits before the start position
	entry &= (ValidityMask::ValidityBuffer::MAX_ENTRY << idx_in_entry);

	// Check the non-ragged entries
	for (const auto entry_count = bits.EntryCount(n); entry_idx < entry_count; ++entry_idx) {
		if (entry) {
			for (; idx_in_entry < bits.BITS_PER_VALUE; ++idx_in_entry, ++j) {
				if (bits.RowIsValid(entry, idx_in_entry)) {
					return j;
				}
			}
		} else {
			j += bits.BITS_PER_VALUE - idx_in_entry;
		}

		entry = bits.GetValidityEntry(entry_idx);
		idx_in_entry = 0;
	}

	// Check the final entry
	for (; j < n; ++idx_in_entry, ++j) {
		if (bits.RowIsValid(entry, idx_in_entry)) {
			return j;
		}
	}

	return j;
}


idx_t IEJoinUnion::JoinComplexBlocks(SelectionVector &lsel, SelectionVector &rsel) {
	// Initialize sweep line if not already done
    if (events.empty()) {
        InitializeSweepline();
    }
    
    idx_t result_count = 0;
    idx_t max_count = STANDARD_VECTOR_SIZE;
    
    // Process events until we have enough results or run out of events
    while (result_count < max_count && current_event < events.size()) {
        const auto& event = events[current_event++];
        int64_t row_id = event.rid;
        bool is_lhs = row_id > 0;
        
        if (event.is_start) {
            // Start event - add to active set and generate results
            if (is_lhs) {
                // Add to active LHS set
                active_lhs.insert(row_id);
                
                // Generate results with all active RHS rows
                for (const auto& rhs_id : active_rhs) {
                    // Skip if we're at capacity - check size
                    if (result_count >= max_count) {
                        // Back up one event so we can continue from here next time
                        current_event--;
                        return result_count;
                    }
                    
                    // Convert to selection vector indices
                    lsel.set_index(result_count, row_id - 1);     // Positive IDs start at 1
                    rsel.set_index(result_count, (-rhs_id) - 1);  // Negative IDs start at -1
                    result_count++;
                }
            } else {
                // Add to active RHS set
                active_rhs.insert(row_id);
                
                // Generate results with all active LHS rows
                for (const auto& lhs_id : active_lhs) {

                    if (result_count >= max_count) {
                        // Back up one event so we can continue from here next time
                        current_event--;
                        return result_count;
                    }
                    
                    // Convert to selection vector indices
                    lsel.set_index(result_count, lhs_id - 1);
                    rsel.set_index(result_count, (-row_id) - 1);
                    result_count++;
                }
            }
        } else {
            // End event - remove from active set
            if (is_lhs) {
                active_lhs.erase(row_id);
            } else {
                active_rhs.erase(row_id);
            }
        }
    }
    fprintf(stderr, "\n result_count = %zu\n", result_count);
    return result_count;
}

/*
idx_t IEJoinUnion::JoinComplexBlocks(SelectionVector &lsel, SelectionVector &rsel) {
	// 8. initialize join result as an empty list for tuple pairs
	idx_t result_count = 0;

	// 11. for(i←1 to n) do
	while (i < n) {
		// 13. for (j ← pos+eqOff to n) do
		for (;;) {
			// 14. if B[j] = 1 then

			//	Use the Bloom filter to find candidate blocks
			while (j < n) {
				auto bloom_begin = NextValid(bloom_filter, j / BLOOM_CHUNK_BITS, bloom_count) * BLOOM_CHUNK_BITS;
				auto bloom_end = MinValue<idx_t>(n, bloom_begin + BLOOM_CHUNK_BITS);

				j = MaxValue<idx_t>(j, bloom_begin);
				j = NextValid(bit_mask, j, bloom_end);
				if (j < bloom_end) {
					break;
				}
			}

			if (j >= n) {
				break;
			}

			// Filter out tuples with the same sign (they come from the same table)
			const auto rrid = li[j];
			++j;

			D_ASSERT(lrid > 0 && rrid < 0);
			// 15. add tuples w.r.t. (L1[j], L1[i]) to join result
			lsel.set_index(result_count, sel_t(+lrid - 1));
			rsel.set_index(result_count, sel_t(-rrid - 1));
			++result_count;
			if (result_count == STANDARD_VECTOR_SIZE) {
				// out of space!
				return result_count;
			}
		}
		++i;

		if (!NextRow()) {
			break;
		}
	}

	return result_count;
}
*/
class IEJoinLocalSourceState : public LocalSourceState {
public:
	explicit IEJoinLocalSourceState(ClientContext &context, const PhysicalIEJoin &op)
	    : op(op), true_sel(STANDARD_VECTOR_SIZE), left_executor(context), right_executor(context),
	      left_matches(nullptr), right_matches(nullptr) {
		auto &allocator = Allocator::Get(context);
		unprojected.Initialize(allocator, op.unprojected_types);

		if (op.conditions.size() < 3) {
			return;
		}

		vector<LogicalType> left_types;
		vector<LogicalType> right_types;
		for (idx_t i = 2; i < op.conditions.size(); ++i) {
			const auto &cond = op.conditions[i];

			left_types.push_back(cond.left->return_type);
			left_executor.AddExpression(*cond.left);

			right_types.push_back(cond.left->return_type);
			right_executor.AddExpression(*cond.right);
		}

		left_keys.Initialize(allocator, left_types);
		right_keys.Initialize(allocator, right_types);
	}

	idx_t SelectOuterRows(bool *matches) {
		idx_t count = 0;
		for (; outer_idx < outer_count; ++outer_idx) {
			if (!matches[outer_idx]) {
				true_sel.set_index(count++, outer_idx);
				if (count >= STANDARD_VECTOR_SIZE) {
					outer_idx++;
					break;
				}
			}
		}

		return count;
	}

	const PhysicalIEJoin &op;

	// Joining
	unique_ptr<IEJoinUnion> joiner;

	idx_t left_base;
	idx_t left_block_index;

	idx_t right_base;
	idx_t right_block_index;

	// Trailing predicates
	SelectionVector true_sel;

	ExpressionExecutor left_executor;
	DataChunk left_keys;

	ExpressionExecutor right_executor;
	DataChunk right_keys;

	DataChunk unprojected;

	// Outer joins
	idx_t outer_idx;
	idx_t outer_count;
	bool *left_matches;
	bool *right_matches;
};

void PhysicalIEJoin::ResolveComplexJoin(ExecutionContext &context, DataChunk &result, LocalSourceState &state_p) const {
	auto &state = state_p.Cast<IEJoinLocalSourceState>();
	auto &ie_sink = sink_state->Cast<IEJoinGlobalState>();
	auto &left_table = *ie_sink.tables[0];
	auto &right_table = *ie_sink.tables[1];

	const auto left_cols = children[0].get().GetTypes().size();
	auto &chunk = state.unprojected;
	do {
		SelectionVector lsel(STANDARD_VECTOR_SIZE);
		SelectionVector rsel(STANDARD_VECTOR_SIZE);
		auto result_count = state.joiner->JoinComplexBlocks(lsel, rsel);
		if (result_count == 0) {
			// exhausted this pair
			return;
		}

		// found matches: extract them

		chunk.Reset();
		SliceSortedPayload(chunk, left_table.global_sort_state, state.left_block_index, lsel, result_count, 0);
		SliceSortedPayload(chunk, right_table.global_sort_state, state.right_block_index, rsel, result_count,
		                   left_cols);
		chunk.SetCardinality(result_count);

		auto sel = FlatVector::IncrementalSelectionVector();
		if (conditions.size() > 2) {
			// If there are more expressions to compute,
			// split the result chunk into the left and right halves
			// so we can compute the values for comparison.
			const auto tail_cols = conditions.size() - 2;

			DataChunk right_chunk;
			chunk.Split(right_chunk, left_cols);
			state.left_executor.SetChunk(chunk);
			state.right_executor.SetChunk(right_chunk);

			auto tail_count = result_count;
			auto true_sel = &state.true_sel;
			for (size_t cmp_idx = 0; cmp_idx < tail_cols; ++cmp_idx) {
				auto &left = state.left_keys.data[cmp_idx];
				state.left_executor.ExecuteExpression(cmp_idx, left);

				auto &right = state.right_keys.data[cmp_idx];
				state.right_executor.ExecuteExpression(cmp_idx, right);

				if (tail_count < result_count) {
					left.Slice(*sel, tail_count);
					right.Slice(*sel, tail_count);
				}
				tail_count = SelectJoinTail(conditions[cmp_idx + 2].comparison, left, right, sel, tail_count, true_sel);
				sel = true_sel;
			}
			chunk.Fuse(right_chunk);

			if (tail_count < result_count) {
				result_count = tail_count;
				chunk.Slice(*sel, result_count);
			}
		}

		//	We need all of the data to compute other predicates,
		//	but we only return what is in the projection map
		ProjectResult(chunk, result);

		// found matches: mark the found matches if required
		if (left_table.found_match) {
			for (idx_t i = 0; i < result_count; i++) {
				left_table.found_match[state.left_base + lsel[sel->get_index(i)]] = true;
			}
		}
		if (right_table.found_match) {
			for (idx_t i = 0; i < result_count; i++) {
				right_table.found_match[state.right_base + rsel[sel->get_index(i)]] = true;
			}
		}
		result.Verify();
	} while (result.size() == 0);
}

class IEJoinGlobalSourceState : public GlobalSourceState {
public:
	explicit IEJoinGlobalSourceState(const PhysicalIEJoin &op, IEJoinGlobalState &gsink)
	    : op(op), gsink(gsink), initialized(false), next_pair(0), completed(0), left_outers(0), next_left(0),
	      right_outers(0), next_right(0) {
	}

	void Initialize() {
		auto guard = Lock();
		if (initialized) {
			return;
		}

		// Compute the starting row for reach block
		// (In theory these are all the same size, but you never know...)
		auto &left_table = *gsink.tables[0];
		const auto left_blocks = left_table.BlockCount();
		idx_t left_base = 0;

		for (size_t lhs = 0; lhs < left_blocks; ++lhs) {
			left_bases.emplace_back(left_base);
			left_base += left_table.BlockSize(lhs);
		}

		auto &right_table = *gsink.tables[1];
		const auto right_blocks = right_table.BlockCount();
		idx_t right_base = 0;
		for (size_t rhs = 0; rhs < right_blocks; ++rhs) {
			right_bases.emplace_back(right_base);
			right_base += right_table.BlockSize(rhs);
		}

		// Outer join block counts
		if (left_table.found_match) {
			left_outers = left_blocks;
		}

		if (right_table.found_match) {
			right_outers = right_blocks;
		}

		// Ready for action
		initialized = true;
	}

public:
	idx_t MaxThreads() override {
		// We can't leverage any more threads than block pairs.
		const auto &sink_state = (op.sink_state->Cast<IEJoinGlobalState>());
		return sink_state.tables[0]->BlockCount() * sink_state.tables[1]->BlockCount();
	}

	void GetNextPair(ClientContext &client, IEJoinLocalSourceState &lstate) {
		auto &left_table = *gsink.tables[0];
		auto &right_table = *gsink.tables[1];

		const auto left_blocks = left_table.BlockCount();
		const auto right_blocks = right_table.BlockCount();
		const auto pair_count = left_blocks * right_blocks;

		// Regular block
		const auto i = next_pair++;
		if (i < pair_count) {
			const auto b1 = i / right_blocks;
			const auto b2 = i % right_blocks;

			lstate.left_block_index = b1;
			lstate.left_base = left_bases[b1];

			lstate.right_block_index = b2;
			lstate.right_base = right_bases[b2];

			lstate.joiner = make_uniq<IEJoinUnion>(client, op, left_table, b1, right_table, b2);
			return;
		}

		// Outer joins
		if (!left_outers && !right_outers) {
			return;
		}

		// Spin wait for regular blocks to finish(!)
		while (completed < pair_count) {
			std::this_thread::yield();
		}

		// Left outer blocks
		const auto l = next_left++;
		if (l < left_outers) {
			lstate.joiner = nullptr;
			lstate.left_block_index = l;
			lstate.left_base = left_bases[l];

			lstate.left_matches = left_table.found_match.get() + lstate.left_base;
			lstate.outer_idx = 0;
			lstate.outer_count = left_table.BlockSize(l);
			return;
		} else {
			lstate.left_matches = nullptr;
		}

		// Right outer block
		const auto r = next_right++;
		if (r < right_outers) {
			lstate.joiner = nullptr;
			lstate.right_block_index = r;
			lstate.right_base = right_bases[r];

			lstate.right_matches = right_table.found_match.get() + lstate.right_base;
			lstate.outer_idx = 0;
			lstate.outer_count = right_table.BlockSize(r);
			return;
		} else {
			lstate.right_matches = nullptr;
		}
	}

	void PairCompleted(ClientContext &client, IEJoinLocalSourceState &lstate) {
		lstate.joiner.reset();
		++completed;
		GetNextPair(client, lstate);
	}

	ProgressData GetProgress() const {
		auto &left_table = *gsink.tables[0];
		auto &right_table = *gsink.tables[1];

		const auto left_blocks = left_table.BlockCount();
		const auto right_blocks = right_table.BlockCount();
		const auto pair_count = left_blocks * right_blocks;

		const auto count = pair_count + left_outers + right_outers;

		const auto l = MinValue(next_left.load(), left_outers.load());
		const auto r = MinValue(next_right.load(), right_outers.load());
		const auto returned = completed.load() + l + r;

		ProgressData res;
		if (count) {
			res.done = double(returned);
			res.total = double(count);
		} else {
			res.SetInvalid();
		}
		return res;
	}

	const PhysicalIEJoin &op;
	IEJoinGlobalState &gsink;

	bool initialized;

	// Join queue state
	atomic<size_t> next_pair;
	atomic<size_t> completed;

	// Block base row number
	vector<idx_t> left_bases;
	vector<idx_t> right_bases;

	// Outer joins
	atomic<idx_t> left_outers;
	atomic<idx_t> next_left;

	atomic<idx_t> right_outers;
	atomic<idx_t> next_right;
};

unique_ptr<GlobalSourceState> PhysicalIEJoin::GetGlobalSourceState(ClientContext &context) const {
	auto &gsink = sink_state->Cast<IEJoinGlobalState>();
	return make_uniq<IEJoinGlobalSourceState>(*this, gsink);
}

unique_ptr<LocalSourceState> PhysicalIEJoin::GetLocalSourceState(ExecutionContext &context,
                                                                 GlobalSourceState &gstate) const {
	return make_uniq<IEJoinLocalSourceState>(context.client, *this);
}

ProgressData PhysicalIEJoin::GetProgress(ClientContext &context, GlobalSourceState &gsource_p) const {
	auto &gsource = gsource_p.Cast<IEJoinGlobalSourceState>();
	return gsource.GetProgress();
}

SourceResultType PhysicalIEJoin::GetData(ExecutionContext &context, DataChunk &result,
                                         OperatorSourceInput &input) const {
	auto &ie_sink = sink_state->Cast<IEJoinGlobalState>();
	auto &ie_gstate = input.global_state.Cast<IEJoinGlobalSourceState>();
	auto &ie_lstate = input.local_state.Cast<IEJoinLocalSourceState>();

	ie_gstate.Initialize();

	if (!ie_lstate.joiner && !ie_lstate.left_matches && !ie_lstate.right_matches) {
		ie_gstate.GetNextPair(context.client, ie_lstate);
	}

	// Process INNER results
	while (ie_lstate.joiner) {
		ResolveComplexJoin(context, result, ie_lstate);

		if (result.size()) {
			return SourceResultType::HAVE_MORE_OUTPUT;
		}

		ie_gstate.PairCompleted(context.client, ie_lstate);
	}

	// Process LEFT OUTER results
	const auto left_cols = children[0].get().GetTypes().size();
	while (ie_lstate.left_matches) {
		const idx_t count = ie_lstate.SelectOuterRows(ie_lstate.left_matches);
		if (!count) {
			ie_gstate.GetNextPair(context.client, ie_lstate);
			continue;
		}
		auto &chunk = ie_lstate.unprojected;
		chunk.Reset();
		SliceSortedPayload(chunk, ie_sink.tables[0]->global_sort_state, ie_lstate.left_block_index, ie_lstate.true_sel,
		                   count);

		// Fill in NULLs to the right
		for (auto col_idx = left_cols; col_idx < chunk.ColumnCount(); ++col_idx) {
			chunk.data[col_idx].SetVectorType(VectorType::CONSTANT_VECTOR);
			ConstantVector::SetNull(chunk.data[col_idx], true);
		}

		ProjectResult(chunk, result);
		result.SetCardinality(count);
		result.Verify();

		return result.size() == 0 ? SourceResultType::FINISHED : SourceResultType::HAVE_MORE_OUTPUT;
	}

	// Process RIGHT OUTER results
	while (ie_lstate.right_matches) {
		const idx_t count = ie_lstate.SelectOuterRows(ie_lstate.right_matches);
		if (!count) {
			ie_gstate.GetNextPair(context.client, ie_lstate);
			continue;
		}

		auto &chunk = ie_lstate.unprojected;
		chunk.Reset();
		SliceSortedPayload(chunk, ie_sink.tables[1]->global_sort_state, ie_lstate.right_block_index, ie_lstate.true_sel,
		                   count, left_cols);

		// Fill in NULLs to the left
		for (idx_t col_idx = 0; col_idx < left_cols; ++col_idx) {
			chunk.data[col_idx].SetVectorType(VectorType::CONSTANT_VECTOR);
			ConstantVector::SetNull(chunk.data[col_idx], true);
		}

		ProjectResult(chunk, result);
		result.SetCardinality(count);
		result.Verify();

		break;
	}

	return result.size() == 0 ? SourceResultType::FINISHED : SourceResultType::HAVE_MORE_OUTPUT;
}

//===--------------------------------------------------------------------===//
// Pipeline Construction
//===--------------------------------------------------------------------===//
void PhysicalIEJoin::BuildPipelines(Pipeline &current, MetaPipeline &meta_pipeline) {
	D_ASSERT(children.size() == 2);
	if (meta_pipeline.HasRecursiveCTE()) {
		throw NotImplementedException("IEJoins are not supported in recursive CTEs yet");
	}

	// becomes a source after both children fully sink their data
	meta_pipeline.GetState().SetPipelineSource(current, *this);

	// Create one child meta pipeline that will hold the LHS and RHS pipelines
	auto &child_meta_pipeline = meta_pipeline.CreateChildMetaPipeline(current, *this);

	// Build out LHS
	auto lhs_pipeline = child_meta_pipeline.GetBasePipeline();
	children[0].get().BuildPipelines(*lhs_pipeline, child_meta_pipeline);

	// Build out RHS
	auto &rhs_pipeline = child_meta_pipeline.CreatePipeline();
	children[1].get().BuildPipelines(rhs_pipeline, child_meta_pipeline);

	// Despite having the same sink, RHS and everything created after it need their own (same) PipelineFinishEvent
	child_meta_pipeline.AddFinishEvent(rhs_pipeline);
}

} // namespace duckdb
