// This is a custom operator to perform an overlap join in DuckDb.
// It is a specialized join which involves creating a sweepline index to do overlap joins on time-series data.
// we are basing this off of src/execution/operator/join/physical_range_join.cpp
// This code will be performance and parallel thread-safety in mind
// but we will not be registering the logical operator yet.

#include "duckdb/execution/operator/join/physical_overlap_join.hpp"

#include "duckdb/common/fast_mem.hpp"
#include "duckdb/common/operator/comparison_operators.hpp"
#include "duckdb/common/row_operations/row_operations.hpp"
#include "duckdb/common/sort/comparators.hpp"
#include "duckdb/common/sort/sort.hpp"
#include "duckdb/common/types/validity_mask.hpp"
#include "duckdb/common/types/vector.hpp"
#include "duckdb/common/vector_operations/vector_operations.hpp"
#include "duckdb/execution/expression_executor.hpp"
#include "duckdb/main/client_context.hpp"
#include "duckdb/parallel/base_pipeline_event.hpp"
#include "duckdb/parallel/thread_context.hpp"
#include "duckdb/parallel/executor_task.hpp"
#include "duckdb/parallel/meta_pipeline.hpp"

#include <thread>

namespace duckdb {

// Sweepline index class for efficient overlap detection
SweeplineIndex::SweeplineIndex() {
    // use sorted vectors
    // standard vector is fine, cache is good
    // don't need to worry about thread-safety for now since it is constructed after
        // This is made in Global Sort Table - singular thread construction for right side
}

void SweeplineIndex::AddInterval(idx_t row_id, timestamp_t start, timestamp_t end) {
    // Add interval to the index
    intervals.push_back({row_id, start, end});

    // add the events to the index
        // I need to know when it starts and ends for this specific row
    events.push_back({start, true, intervals.size() - 1});
    events.push_back({end, false, intervals.size() - 1});
}

// Needs to be used!!!e
    // DuckDB > Sort > Local > Global > Create Index(right) > Prepare(Sort) > Run overlap
void SweeplineIndex::Prepare() {
    // Prepare the index for searching
    // Sort intervals by start time
    std::sort(events.begin(), events.end(), [](const Event &a, const Event &b) {
        if (a.time != b.time) {
            return a.time < b.time;
        }

        // if equal time then get end events
        return a.is_start < b.is_start;
    });
}

// couple of strategies
    // batching
    // iteratively
    // whole chunk
        // for simplicity we are going back to whole chunk and allowing the scheduler to handle this
        // 04-16: change of plans again
            // provide input of start/end and process those
void SweeplineIndex::FindOverlaps(
                PhysicalOverlapJoin::GlobalSortedTable &left_table, 
                PhysicalOverlapJoin::GlobalSortedTable &right_table,
                vector<pair<idx_t, idx_t>> &result_pairs, 
                size_t &current_event_idx,
                unordered_set<idx_t> &active_intervals,
                idx_t start_idx,
                idx_t end_idx) {
                    
    // reset the intervals for this thread
                    // originally wanted to make this sequential but don't know how
                    // make the partitioning sequential
    active_intervals.clear();

    if (end_idx <= start_idx) {
        return; // No rows to process
    }
    
    // Get the range of rows to process from left_table
        // change to start_idx and end_idx
    /*idx_t start_idx = 0;
    idx_t end_idx = left_table.Count();*/

    // check if out of bounds
    end_idx = std::min(end_idx, left_table.Count());
    
    // Since the left table is sorted by start time, we can process events incrementally
    size_t event_idx = current_event_idx;
    
    for (idx_t left_idx = start_idx; left_idx < end_idx; left_idx++) {
        // Get start/end times directly using your pointer methods
        timestamp_t query_start = left_table.GetStartTime(left_idx);
        timestamp_t query_end = left_table.GetEndTime(left_idx);
        
        // Advance the sweepline to the current query start time
        while (event_idx < events.size() && events[event_idx].time < query_start) {
            // Update active intervals set
            if (events[event_idx].is_start) {
                active_intervals.insert(events[event_idx].interval_idx);
            } else {
                active_intervals.erase(events[event_idx].interval_idx);
            }
            event_idx++;
        }
        
        // Check active intervals for overlaps
        for (idx_t interval_idx : active_intervals) {
            const Interval& interval = intervals[interval_idx];
            if (interval.end >= query_start) {
                result_pairs.emplace_back(left_idx, interval.row_id);
            }
        }
        
        // Process events between query_start and query_end
        size_t temp_event_idx = event_idx;
        while (temp_event_idx < events.size() && events[temp_event_idx].time <= query_end) {
            const Event& event = events[temp_event_idx];
            if (event.is_start) {
                // A new interval starts during our query range
                const Interval& interval = intervals[event.interval_idx];
                if (interval.end >= query_start) {
                    result_pairs.emplace_back(left_idx, interval.row_id);
                }
            }
            temp_event_idx++;
        }
    }
    
    // Save the current event index for efficiency
    current_event_idx = event_idx;
}


//===--------------------------------------------------------------------===//
// LocalSortedTable Implementation
//===--------------------------------------------------------------------===//
//-- 123
PhysicalOverlapJoin::LocalSortedTable::LocalSortedTable(ClientContext &context, const PhysicalOverlapJoin &op,
                                                      const idx_t child)
    : op(op), executor(context), has_null(0), count(0) {
	// Initialize order clause expression executor and key DataChunk
	vector<LogicalType> types;
	for (const auto &cond : op.conditions) {
		const auto &expr = child ? cond.right : cond.left;
		executor.AddExpression(*expr);

		types.push_back(expr->return_type);
	}
	auto &allocator = Allocator::Get(context);
	keys.Initialize(allocator, types);
}

// -- 123
    // Sink can be almost identical - what about sorting by timestamp

    // local table sink
void PhysicalOverlapJoin::LocalSortedTable::Sink(DataChunk &input, GlobalSortState &global_sort_state) {
	// Initialize local state (if necessary)
        // lazy initialization
	if (!local_sort_state.initialized) {
		local_sort_state.Initialize(global_sort_state, global_sort_state.buffer_manager);
	}

	// Obtain sorting columns
	keys.Reset();
    // execute goes and fills the keys based on the input from SQL after the parser
        // no need to change this - will automatically identify the time columns (start,end)
	executor.Execute(input, keys);

	// Do not operate on primary key directly to avoid modifying the input chunk
	Vector primary = keys.data[0];
	// Count the NULLs so we can exclude them later
	has_null += MergeNulls(primary, op.conditions);
	count += keys.size();

	//	Only sort the primary key
	DataChunk join_head;
	join_head.data.emplace_back(primary);
	join_head.SetCardinality(keys.size());

	// Sink the data into the local sort state
	local_sort_state.SinkChunk(join_head, input);
}

// -- 987
    // need to check on this one
idx_t PhysicalOverlapJoin::LocalSortedTable::MergeNulls(Vector &primary, const vector<JoinCondition> &conditions) {
    // Implementation of MergeNulls method
        // don't know if this is required
    return 0; // Placeholder - don't even need to do anything
}

//===--------------------------------------------------------------------===//
// GlobalSortedTable Implementation
//===--------------------------------------------------------------------===//

// do I need buffer manager here?
PhysicalOverlapJoin::GlobalSortedTable::GlobalSortedTable(ClientContext &context, const vector<BoundOrderByNode> &orders,
                                                        RowLayout &payload_layout, const PhysicalOperator &op_p)
    : op(op_p), global_sort_state(context, orders, payload_layout), has_null(0), count(0), memory_per_thread(0) {

	// Set external (can be forced with the PRAGMA)
    fprintf(stderr, "OverlapJoin: GetData called\n");
	auto &config = ClientConfig::GetConfig(context);
	global_sort_state.external = config.force_external;
	memory_per_thread = PhysicalOverlapJoin::GetMaxThreadMemory(context);
}

// had an issue with passing const to the global_sort_state
timestamp_t PhysicalOverlapJoin::GlobalSortedTable::GetStartTime(idx_t row_idx) {
    // Access the sorted block
    D_ASSERT(!global_sort_state.sorted_blocks.empty());
    auto &block = global_sort_state.sorted_blocks[0];
    auto &layout = block->payload_data->layout;
    
    // Calculate the offset for the start time column
    idx_t start_time_offset = layout.GetOffsets()[0]; // Assuming start time is the first column
    
    // Access the row data
    SBScanState scan_state(global_sort_state.buffer_manager, global_sort_state);
    scan_state.sb = block.get();
    scan_state.SetIndices(0, row_idx);
    scan_state.PinData(*block->payload_data);
    
    // Get a pointer to the row data
    auto data_ptr = scan_state.DataPtr(*block->payload_data);
    
    // Important: Don't add row_idx * layout.GetRowWidth() again since SetIndices already positions us
    // at the correct row
    return *((timestamp_t *)(data_ptr + start_time_offset));
}

timestamp_t PhysicalOverlapJoin::GlobalSortedTable::GetEndTime(idx_t row_idx) {
    // Access the sorted block
    D_ASSERT(!global_sort_state.sorted_blocks.empty());
    auto &block = global_sort_state.sorted_blocks[0];
    auto &layout = block->payload_data->layout;
    
    // Calculate the offset for the end time column
    idx_t end_time_offset = layout.GetOffsets()[1]; // Assuming end time is the second column
    
    // Access the row data
    SBScanState scan_state(global_sort_state.buffer_manager, global_sort_state);
    scan_state.sb = block.get();
    scan_state.SetIndices(0, row_idx);
    scan_state.PinData(*block->payload_data);
    
    // Get a pointer to the row data
    auto data_ptr = scan_state.DataPtr(*block->payload_data);
    
    // Just use the offset without adding row_idx calculations since SetIndices handles that
    return *((timestamp_t *)(data_ptr + end_time_offset));
}
void PhysicalOverlapJoin::GlobalSortedTable::Combine(LocalSortedTable &ltable) {
	global_sort_state.AddLocalState(ltable.local_sort_state);
	has_null += ltable.has_null;
	count += ltable.count;
}

void PhysicalOverlapJoin::GlobalSortedTable::BuildSweeplineIndex() {
    // Implementation of BuildSweeplineIndex
        // make_unique is deprecated, use make_uniq
    sweep_index = make_uniq<SweeplineIndex>();

    // still need to use the helper functions to get start and end
    for(idx_t row_idx = 0; row_idx < Count() ; row_idx++) {
        auto start_time = GetStartTime(row_idx);
        auto end_time = GetEndTime(row_idx);

        // add to Intervals
        sweep_index->AddInterval(row_idx, start_time, end_time);
    }

    // run prepare for global merge
    sweep_index->Prepare();
    
    // Build the index from sorted data
}

// outputs the current state
void PhysicalOverlapJoin::GlobalSortedTable::Print() {
    global_sort_state.Print();
}

//===--------------------------------------------------------------------===//
// Task Classes for Parallel Processing of the GLOBAL SORT/MERGE -- NOT OPERATOR
//===--------------------------------------------------------------------===//
class OverlapJoinMergeTask : public ExecutorTask {
public:
    using GlobalSortedTable = PhysicalOverlapJoin::GlobalSortedTable;

public:
    OverlapJoinMergeTask(shared_ptr<Event> event_p, ClientContext &context, GlobalSortedTable &table)
        : ExecutorTask(context, std::move(event_p), table.op), context(context), table(table) {
    }

	TaskExecutionResult ExecuteTask(TaskExecutionMode mode) override {
		// Initialize iejoin sorted and iterate until done
            // this is just sorting
            // no merge is happenig here yet - prepping global sort table
		auto &global_sort_state = table.global_sort_state;
		MergeSorter merge_sorter(global_sort_state, BufferManager::GetBufferManager(context));
		merge_sorter.PerformInMergeRound();
		event->FinishTask();

		return TaskExecutionResult::TASK_FINISHED;
	}
private:
    ClientContext &context;
    GlobalSortedTable &table;
};

class OverlapJoinMergeEvent : public BasePipelineEvent {
public:
    using GlobalSortedTable = PhysicalOverlapJoin::GlobalSortedTable;

public:
    OverlapJoinMergeEvent(GlobalSortedTable &table_p, Pipeline &pipeline_p)
        : BasePipelineEvent(pipeline_p), table(table_p) {
    }

    GlobalSortedTable &table;

public:
    void Schedule() override {
        auto &context = pipeline->GetClientContext();

        // Schedule tasks equal to the number of threads, which will each merge multiple partitions
        auto &ts = TaskScheduler::GetScheduler(context);
        auto num_threads = NumericCast<idx_t>(ts.NumberOfThreads());

        vector<shared_ptr<Task>> iejoin_tasks;
        for (idx_t tnum = 0; tnum < num_threads; tnum++) {
            iejoin_tasks.push_back(make_uniq<OverlapJoinMergeTask>(shared_from_this(), context, table));
        }
        SetTasks(std::move(iejoin_tasks));
    }

    void FinishEvent() override {
		auto &global_sort_state = table.global_sort_state;

		global_sort_state.CompleteMergeRound(true);
		if (global_sort_state.sorted_blocks.size() > 1) {
			// Multiple blocks remaining: Schedule the next round
			table.ScheduleMergeTasks(*pipeline, *this);
		}
    }
};

// this is just the global merge, no operator merge
void PhysicalOverlapJoin::GlobalSortedTable::ScheduleMergeTasks(Pipeline &pipeline, Event &event) {
    // init global sort state for merge
	global_sort_state.InitializeMergeRound();
	auto new_event = make_shared_ptr<OverlapJoinMergeEvent>(*this, pipeline);
	event.InsertEvent(std::move(new_event));
}

// this is just global sort finalize, not operator finalize
void PhysicalOverlapJoin::GlobalSortedTable::Finalize(Pipeline &pipeline, Event &event) {
	// Prepare for merge sort phase
        // THIS IS GLOBALSORTTABLE FINALIZE - NOT JOIN FINALIZE
	global_sort_state.PrepareMergePhase();

	// Start the merge phase or finish if a merge is not necessary
	if (global_sort_state.sorted_blocks.size() > 1) {
		ScheduleMergeTasks(pipeline, event);
	}
}

BufferHandle PhysicalOverlapJoin::SliceSortedPayload(DataChunk &payload, GlobalSortState &state, const idx_t block_idx,
                                                   const SelectionVector &result, const idx_t result_count,
                                                   const idx_t left_cols) {
    D_ASSERT(state.sorted_blocks.size() == 1);
	SBScanState read_state(state.buffer_manager, state);
	read_state.sb = state.sorted_blocks[0].get();
	auto &sorted_data = *read_state.sb->payload_data;

	read_state.SetIndices(block_idx, 0);
	read_state.PinData(sorted_data);
	const auto data_ptr = read_state.DataPtr(sorted_data);
	data_ptr_t heap_ptr = nullptr;

	// Set up a batch of pointers to scan data from
	Vector addresses(LogicalType::POINTER, result_count);
	auto data_pointers = FlatVector::GetData<data_ptr_t>(addresses);

	// Set up the data pointers for the values that are actually referenced
	const idx_t &row_width = sorted_data.layout.GetRowWidth();

	auto prev_idx = result.get_index(0);
	SelectionVector gsel(result_count);
	idx_t addr_count = 0;
	gsel.set_index(0, addr_count);
	data_pointers[addr_count] = data_ptr + prev_idx * row_width;
	for (idx_t i = 1; i < result_count; ++i) {
		const auto row_idx = result.get_index(i);
		if (row_idx != prev_idx) {
			data_pointers[++addr_count] = data_ptr + row_idx * row_width;
			prev_idx = row_idx;
		}
		gsel.set_index(i, addr_count);
	}
	++addr_count;

	// Unswizzle the offsets back to pointers (if needed)
	if (!sorted_data.layout.AllConstant() && state.external) {
		heap_ptr = read_state.payload_heap_handle.Ptr();
	}

	// Deserialize the payload data
	auto sel = FlatVector::IncrementalSelectionVector();
	for (idx_t col_no = 0; col_no < sorted_data.layout.ColumnCount(); col_no++) {
		auto &col = payload.data[left_cols + col_no];
		RowOperations::Gather(addresses, *sel, col, *sel, addr_count, sorted_data.layout, col_no, 0, heap_ptr);
		col.Slice(gsel, result_count);
	}

    // BufferHandle
	return std::move(read_state.payload_heap_handle);
    //return BufferHandle();
}

/* - redundant FindOverlaps call


    // this is the helper method for running FindOverlaps on the global left and right sides
void PhysicalOverlapJoin::FindOverlaps(GlobalSortedTable &left, GlobalSortedTable &right,
                                      vector<pair<idx_t, idx_t>> &result_pairs) const {
    // Implementation of FindOverlaps for left and right side of Sorted Table
}

*/


// should be no change since the projection is not overlap join specific
void PhysicalOverlapJoin::ProjectResult(DataChunk &chunk, DataChunk &result) const {
	const auto left_projected = left_projection_map.size();
	for (idx_t i = 0; i < left_projected; ++i) {
		result.data[i].Reference(chunk.data[left_projection_map[i]]);
	}
	const auto left_width = children[0].get().GetTypes().size();
	for (idx_t i = 0; i < right_projection_map.size(); ++i) {
		result.data[left_projected + i].Reference(chunk.data[left_width + right_projection_map[i]]);
	}
	result.SetCardinality(chunk);
}

// !!! TODO: !!!
PhysicalOverlapJoin::PhysicalOverlapJoin(LogicalComparisonJoin &op, PhysicalOperator &left,
                                            PhysicalOperator &right, vector<JoinCondition> cond, JoinType join_type,
                                            idx_t estimated_cardinality)
        : PhysicalComparisonJoin(
                op, 
                PhysicalOperatorType::OVERLAP_JOIN, 
                std::move(cond), 
                join_type, 
                estimated_cardinality) 
    {
    // pass children to state
    children.push_back(left);
    children.push_back(right);
        
    // Fill out the left projection map.
    left_projection_map = op.left_projection_map;
    if (left_projection_map.empty()) {
        const auto left_count = children[0].get().GetTypes().size();
        left_projection_map.reserve(left_count);
        for (column_t i = 0; i < left_count; ++i) {
            left_projection_map.emplace_back(i);
        }
    }
    
    // Fill out the right projection map.
    right_projection_map = op.right_projection_map;
    if (right_projection_map.empty()) {
        const auto right_count = children[1].get().GetTypes().size();
        right_projection_map.reserve(right_count);
        for (column_t i = 0; i < right_count; ++i) {
            right_projection_map.emplace_back(i);
        }
    }
    
    // Extract types
        // well it's only timestamp - so I don't care about it
    /*
    for (auto &condition : conditions) {
        join_key_types.push_back(condition.left->return_type);
    }*/
    
    // since we're using pragma to force the operator this is just for testing
    D_ASSERT(conditions.size() >= 2);
    
    // We only care about ordering the start time which is conditions[0]
    lhs_orders.emplace_back(
        OrderType::ASCENDING,
        OrderByNullType::NULLS_LAST,
        conditions[0].left->Copy()
    );
    
    rhs_orders.emplace_back(
        OrderType::ASCENDING,
        OrderByNullType::NULLS_LAST,
        conditions[0].right->Copy()
    );

}

//===--------------------------------------------------------------------===//
    // Sink
//===--------------------------------------------------------------------===//

// reuse
class OverlapJoinLocalState : public LocalSinkState {
    public:
        using LocalSortedTable = PhysicalOverlapJoin::LocalSortedTable;
    
        OverlapJoinLocalState(ClientContext &context, const PhysicalOverlapJoin &op, const idx_t child)
            : table(context, op, child) {
        }
    
        //! The local sort state
        LocalSortedTable table;
    };

class OverlapGlobalState : public GlobalSinkState {
    public:
        // borrow from range join's global state
        using GlobalSortedTable = PhysicalOverlapJoin::GlobalSortedTable;
    
    public:
        // table[0] is left and table[1] is right
        OverlapGlobalState(ClientContext &context, const PhysicalOverlapJoin &op) : child(0) {
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
    
        // this is for passing between different phases of iejoin
            // keep it for when processing different chunks

        OverlapGlobalState(OverlapGlobalState &prev) : tables(std::move(prev.tables)), child(prev.child + 1) {
            state = prev.state;
        }
    
        void Sink(DataChunk &input, OverlapJoinLocalState &lstate) {
            //auto &local_table = tables[child];
            auto &global_table = *tables[child];
            auto &global_sort_state = global_table.global_sort_state;
            auto &local_sort_state = lstate.table.local_sort_state;
    
            // Sink the data into the local sort state
            lstate.table.Sink(input, global_sort_state);
    
            // When sorting data reaches a certain size, we sort it
            if (local_sort_state.SizeInBytes() >= global_table.memory_per_thread) {
                local_sort_state.Sort(global_sort_state, true);
            }
        }
    
        // variables of GlobalSortedTable
        vector<unique_ptr<GlobalSortedTable>> tables;
        size_t child;
    };


unique_ptr<GlobalSinkState> PhysicalOverlapJoin::GetGlobalSinkState(ClientContext &context) const {
	D_ASSERT(!sink_state);
	return make_uniq<OverlapGlobalState>(context, *this);
}

unique_ptr<LocalSinkState> PhysicalOverlapJoin::GetLocalSinkState(ExecutionContext &context) const {
	idx_t sink_child = 0;
	if (sink_state) {
		const auto &ie_sink = sink_state->Cast<OverlapGlobalState>();
		sink_child = ie_sink.child;
	}
	return make_uniq<OverlapJoinLocalState>(context.client, *this, sink_child);
}

SinkResultType PhysicalOverlapJoin::Sink(ExecutionContext &context, DataChunk &chunk, OperatorSinkInput &input) const {
	auto &gstate = input.global_state.Cast<OverlapGlobalState>();
	auto &lstate = input.local_state.Cast<OverlapJoinLocalState>();

	gstate.Sink(chunk, lstate);

	return SinkResultType::NEED_MORE_INPUT;
}

SinkCombineResultType PhysicalOverlapJoin::Combine(ExecutionContext &context, OperatorSinkCombineInput &input) const {
	auto &gstate = input.global_state.Cast<OverlapGlobalState>();
	auto &lstate = input.local_state.Cast<OverlapJoinLocalState>();
	gstate.tables[gstate.child]->Combine(lstate.table);
	auto &client_profiler = QueryProfiler::Get(context.client);

	context.thread.profiler.Flush(*this);
	client_profiler.Flush(context.thread.profiler);

	return SinkCombineResultType::FINISHED;
}

SinkFinalizeType PhysicalOverlapJoin::Finalize(Pipeline &pipeline, Event &event, ClientContext &context,
    OperatorSinkFinalizeInput &input) const {
    auto &gstate = input.global_state.Cast<OverlapGlobalState>();
    auto &table = *gstate.tables[gstate.child];
    auto &global_sort_state = table.global_sort_state;

    // Handle empty input on the RHS
    if (gstate.child == 1 && global_sort_state.sorted_blocks.empty()) {
        return SinkFinalizeType::NO_OUTPUT_POSSIBLE;
    }

    // Finalize sorting for the current child
    table.Finalize(pipeline, event);

    // buildsweepline here
        // this way when we call getData we don't have to worry about it
    if (gstate.child == 1) {
        auto &right_table = *gstate.tables[1];
        right_table.BuildSweeplineIndex();
    }

    // Move to the next child (if applicable)
    ++gstate.child;

    return SinkFinalizeType::READY;

}

//===--------------------------------------------------------------------===//
// Operator
//===--------------------------------------------------------------------===//

// shit, executeInternal is not for Sink
    // this is for the volcano pull model
    // I knew i should've stuck more closely to iejoin...
OperatorResultType PhysicalOverlapJoin::ExecuteInternal(ExecutionContext &context, DataChunk &input, DataChunk &chunk,
                                                        GlobalOperatorState &gstate, OperatorState &state) const {
    
    // just leave it as finished, we aren't doing volcano
    return OperatorResultType::FINISHED;                                                        
}

//===--------------------------------------------------------------------===//
// Source
//===--------------------------------------------------------------------===//
class OverlapGlobalSourceState : public GlobalSourceState {
public:
    explicit OverlapGlobalSourceState(OverlapGlobalState &sink_state) {
        // Reference tables from the sink phase
        left_table = sink_state.tables[0].get();
        right_table = sink_state.tables[1].get();
        // Initialize the sweepline index from the right table - creates pointer
        sweepline_index = right_table->sweep_index.get();
        // Initialize the current chunk index
        current_chunk_idx = 0;
    }

    //! Reference to the left and right tables
    PhysicalOverlapJoin::GlobalSortedTable *left_table;
    PhysicalOverlapJoin::GlobalSortedTable *right_table;

    //! Reference to the sweepline index
    SweeplineIndex *sweepline_index;

    //! Current chunk being processed
    atomic<idx_t> current_chunk_idx;

    idx_t MaxThreads() override {
        // Allow multiple threads based on the number of chunks in the left table
        return (left_table->Count() + STANDARD_VECTOR_SIZE - 1) / STANDARD_VECTOR_SIZE;
    }
};

class OverlapLocalSourceState : public LocalSourceState {
public:
    OverlapLocalSourceState() : 
        current_position(0),
        thread_event_idx(0) {
    }

    //! Current position in the chunk being processed
    idx_t current_position;

    //! Sweep-line algorithm state
    size_t thread_event_idx;
    unordered_set<idx_t> thread_active_intervals;

    //! Buffer for holding result pairs
    vector<pair<idx_t, idx_t>> result_pairs;

    //! Reset the local state for the next chunk
    void Reset() {
        current_position = 0;
        thread_event_idx = 0;
        thread_active_intervals.clear();
        result_pairs.clear();
    }
};

unique_ptr<GlobalSourceState> PhysicalOverlapJoin::GetGlobalSourceState(ClientContext &context) const {
    // Get the sink state
    auto &sink_state = static_cast<OverlapGlobalState&>(*this->sink_state);
    // Create global source state that references the sink state
    return make_uniq<OverlapGlobalSourceState>(sink_state);
}

unique_ptr<LocalSourceState> PhysicalOverlapJoin::GetLocalSourceState(ExecutionContext &context,
                                                                      GlobalSourceState &gstate) const {
    return make_uniq<OverlapLocalSourceState>();
}



SourceResultType PhysicalOverlapJoin::GetData(ExecutionContext &context, DataChunk &result,
                                              OperatorSourceInput &input) const {
    auto &gstate = input.global_state.Cast<OverlapGlobalSourceState>();
    auto &lstate = input.local_state.Cast<OverlapLocalSourceState>();

    // If there are no more results, return FINISHED
    idx_t chunk_idx = gstate.current_chunk_idx.fetch_add(1);

    // compare and see if there is anything left from the left table
    if (chunk_idx * STANDARD_VECTOR_SIZE >= gstate.left_table->Count()) {
        return SourceResultType::FINISHED;
    }

    //D_ASSERT()
    // clear out the last batch of results that this thread had
    lstate.result_pairs.clear();

    // run Find_Overlaps - this will go over the whole chunk
    gstate.sweepline_index->FindOverlaps(*gstate.left_table, 
                                        *gstate.right_table, 
                                        lstate.result_pairs, 
                                        lstate.thread_event_idx,
                                        lstate.thread_active_intervals,
                                        chunk_idx * STANDARD_VECTOR_SIZE,
                                        std::min((chunk_idx + 1) * STANDARD_VECTOR_SIZE, gstate.left_table->Count())
                                    );
    
    if (lstate.result_pairs.empty()) {
        // No results found, return FINISHED
        return SourceResultType::FINISHED;
    }

    // check types of the children from left and right
    // should be a non-issue? we only have time
    vector<LogicalType> result_types = children[0].get().GetTypes();
    const auto &right_types = children[1].get().GetTypes();
    for (auto &type : right_types) {
        result_types.push_back(type);
    }

    // initialize the result chunk
    // see duckdb/common/types/data_chunk.hpp
    result.Initialize(Allocator::Get(context.client), result_types);

    idx_t result_count = std::min(idx_t(lstate.result_pairs.size()), (idx_t)STANDARD_VECTOR_SIZE);

    // hold the values of from our result pairs
    SelectionVector left_sel(result_count);
    SelectionVector right_sel(result_count);

    for (idx_t i = 0; i < result_count; i++) {
        left_sel.set_index(i, lstate.result_pairs[i].first);
        right_sel.set_index(i, lstate.result_pairs[i].second);
    }

    DataChunk left_chunk, right_chunk;
    left_chunk.Initialize(Allocator::Get(context.client), children[0].get().GetTypes());
    right_chunk.Initialize(Allocator::Get(context.client), children[1].get().GetTypes());
    
    // Fetch data from the sorted tables
    BufferHandle left_handle = SliceSortedPayload(left_chunk, gstate.left_table->global_sort_state, 0, 
                                                left_sel, result_count);
    BufferHandle right_handle = SliceSortedPayload(right_chunk, gstate.right_table->global_sort_state, 0,
                                                right_sel, result_count);
    result.SetCardinality(result_count);

    // Copy data from left and right chunks to the result chunk
    for (idx_t i = 0; i < left_chunk.ColumnCount(); i++) {
        result.data[i].Reference(left_chunk.data[i]);
    }
    
    for (idx_t i = 0; i < right_chunk.ColumnCount(); i++) {
        result.data[left_chunk.ColumnCount() + i].Reference(right_chunk.data[i]);
    }

    return SourceResultType::HAVE_MORE_OUTPUT;
}

//===--------------------------------------------------------------------===//
// Pipeline Construction
//===--------------------------------------------------------------------===//
void PhysicalOverlapJoin::BuildPipelines(Pipeline &current, MetaPipeline &meta_pipeline) {
	D_ASSERT(children.size() == 2);
	if (meta_pipeline.HasRecursiveCTE()) {
		throw NotImplementedException("OverlapJoins are not supported in recursive CTEs yet");
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