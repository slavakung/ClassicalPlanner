#pragma once

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <memory_resource>
#include <optional>
#include <stdexcept>
#include <utility>
#include <vector>

namespace planning
{
    // Lightweight index handles are intentionally used instead of shared_ptr.
    // They are cheap to copy and keep ancestry storage compact.
    struct PathId {
        std::uint64_t value = invalidValue();

        static consteval std::uint64_t invalidValue() {
            return std::numeric_limits<std::uint64_t>::max();
        }

        [[nodiscard]] constexpr bool valid() const noexcept {
            return value != invalidValue();
        }

        friend constexpr bool operator==(PathId, PathId) = default;
    };

    struct NodeId {
        std::uint64_t value = invalidValue();

        static consteval std::uint64_t invalidValue() {
            return std::numeric_limits<std::uint64_t>::max();
        }

        [[nodiscard]] constexpr bool valid() const noexcept {
            return value != invalidValue();
        }

        friend constexpr bool operator==(NodeId, NodeId) = default;
    };

    enum class NodeStatus {
        open,
        goal,
        deadEnd,
        pruned,
        failed
    };

    // Recyclable path arena. Each new path owns one reference, and each child
    // keeps its parent alive. Release the owner's reference after expansion or
    // recursive return; completed branches then disappear without losing the
    // ancestry needed by pending descendants. It never stores State.
    //
    // Each chunk is SoA (structure-of-arrays): scans over depths or parents do
    // not drag actions/history data through cache unnecessarily.
    template<typename Action, typename HistoryKey>
    class PathArena {
    public:
        explicit PathArena(
            std::pmr::memory_resource* resource,
            std::size_t chunkCapacity = 4096
        )
            : resource_(resource),
              chunkCapacity_(std::max<std::size_t>(1, chunkCapacity))
        {
        }

        [[nodiscard]] PathId AddRoot(
            HistoryKey historyKey
        ) {
            return Add(
                PathId{},
                std::nullopt,
                std::move(historyKey),
                0
            );
        }

        [[nodiscard]] PathId AddChild(
            PathId parent,
            Action action,
            HistoryKey historyKey
        ) {
            const auto parentDepth = Depth(parent);
            if (parentDepth == std::numeric_limits<std::uint64_t>::max()) {
                throw std::overflow_error("path depth overflow");
            }
            return Add(
                parent,
                std::optional<Action>{std::move(action)},
                std::move(historyKey),
                parentDepth + 1
            );
        }

        [[nodiscard]] PathId Parent(PathId id) const {
            return chunkFor(id).parents[offsetOf(id)];
        }

        [[nodiscard]] const std::optional<Action>& ActionFromParent(
            PathId id
        ) const {
            return chunkFor(id).actions[offsetOf(id)];
        }

        [[nodiscard]] const HistoryKey& History(PathId id) const {
            return *chunkFor(id).historyKeys[offsetOf(id)];
        }

        [[nodiscard]] std::uint64_t Depth(PathId id) const {
            return chunkFor(id).depths[offsetOf(id)];
        }

        // Number of live paths, including ancestors retained by children.
        [[nodiscard]] std::size_t size() const noexcept {
            return size_;
        }

        [[nodiscard]] std::size_t capacity() const noexcept {
            return chunks_.size() * chunkCapacity_;
        }

        // PathId copies are borrowed handles, not additional owners. A released
        // handle must not be used again: its slot can belong to a different path.
        // Link free slots through their parent field so releasing a long chain
        // needs neither recursion nor an allocation from the monotonic resource.
        void Release(PathId id) noexcept {
            while (id.valid()) {
                auto& chunk = chunks_[chunkIndex(id)];
                const auto offset = offsetOf(id);
                if (--chunk.references[offset] != 0) return;
                const auto parent = chunk.parents[offset];
                chunk.actions[offset].reset();
                chunk.historyKeys[offset].reset();
                chunk.parents[offset] = freeHead_;
                freeHead_ = id;
                --size_;
                id = parent;
            }
        }

    private:
        struct Chunk {
            std::pmr::vector<PathId> parents;
            std::pmr::vector<std::optional<Action>> actions;
            std::pmr::vector<std::optional<HistoryKey>> historyKeys;
            std::pmr::vector<std::uint64_t> depths;
            std::pmr::vector<std::uint64_t> references;

            Chunk(
                std::pmr::memory_resource* resource,
                std::size_t capacity
            )
                : parents(resource),
                  actions(resource),
                  historyKeys(resource),
                  depths(resource),
                  references(resource)
            {
                // reserve() obtains contiguous backing memory once per field.
                parents.reserve(capacity);
                actions.reserve(capacity);
                historyKeys.reserve(capacity);
                depths.reserve(capacity);
                references.reserve(capacity);
            }
        };

        std::pmr::memory_resource* resource_;
        std::size_t chunkCapacity_;
        std::vector<Chunk> chunks_;
        std::size_t size_ = 0;
        std::size_t nextUnused_ = 0;
        PathId freeHead_;

        [[nodiscard]] std::size_t chunkIndex(PathId id) const {
            return static_cast<std::size_t>(id.value / chunkCapacity_);
        }

        [[nodiscard]] std::size_t offsetOf(PathId id) const {
            return static_cast<std::size_t>(id.value % chunkCapacity_);
        }

        [[nodiscard]] const Chunk& chunkFor(PathId id) const {
            if (!id.valid() || id.value >= nextUnused_) {
                throw std::out_of_range("invalid PathId");
            }
            const auto& chunk = chunks_.at(chunkIndex(id));
            if (chunk.references[offsetOf(id)] == 0) {
                throw std::out_of_range("released PathId");
            }
            return chunk;
        }

        [[nodiscard]] PathId Add(
            PathId parent,
            std::optional<Action> action,
            HistoryKey historyKey,
            std::uint64_t depth
        ) {
            if (parent.valid() && chunkFor(parent).references[offsetOf(parent)] ==
                    std::numeric_limits<std::uint64_t>::max()) {
                throw std::overflow_error("path reference count overflow");
            }
            if (freeHead_.valid()) {
                const auto id = freeHead_;
                auto& chunk = chunks_[chunkIndex(id)];
                const auto offset = offsetOf(id);
                try {
                    if (action) chunk.actions[offset].emplace(std::move(*action));
                    chunk.historyKeys[offset].emplace(std::move(historyKey));
                } catch (...) {
                    chunk.actions[offset].reset();
                    chunk.historyKeys[offset].reset();
                    throw;
                }
                freeHead_ = chunk.parents[offset];
                chunk.parents[offset] = parent;
                chunk.depths[offset] = depth;
                chunk.references[offset] = 1;
                if (parent.valid()) {
                    ++chunks_[chunkIndex(parent)].references[offsetOf(parent)];
                }
                ++size_;
                return id;
            }
            if (nextUnused_ == PathId::invalidValue()) {
                throw std::length_error("path handle capacity exhausted");
            }
            // A failed element construction can leave an empty, reserved chunk.
            // Reuse it on retry instead of shifting the index-to-chunk mapping.
            if (nextUnused_ / chunkCapacity_ == chunks_.size()) {
                chunks_.emplace_back(resource_, chunkCapacity_);
            }

            Chunk& chunk = chunks_.back();
            const auto previousSize = chunk.parents.size();
            try {
                chunk.parents.push_back(parent);
                chunk.actions.push_back(std::move(action));
                chunk.historyKeys.emplace_back(std::move(historyKey));
                chunk.depths.push_back(depth);
                chunk.references.push_back(1);
            } catch (...) {
                // Keep all SoA fields aligned even when a user Action or
                // HistoryKey move constructor throws.
                if (chunk.parents.size() > previousSize) chunk.parents.pop_back();
                if (chunk.actions.size() > previousSize) chunk.actions.pop_back();
                if (chunk.historyKeys.size() > previousSize) chunk.historyKeys.pop_back();
                if (chunk.depths.size() > previousSize) chunk.depths.pop_back();
                if (chunk.references.size() > previousSize) chunk.references.pop_back();
                throw;
            }

            if (parent.valid()) {
                ++chunks_[chunkIndex(parent)].references[offsetOf(parent)];
            }
            ++size_;
            return PathId{static_cast<std::uint64_t>(nextUnused_++)};
        }
    };

    // Recyclable full-state node pool. Nodes are temporary:
    // once expanded, a DFS node can be released and its slot reused. For BFS we
    // release an entire level after expansion. This avoids retaining every full
    // ancestor State while still using contiguous chunks.
    template<typename State, typename Cost>
    class NodePool {
    public:
        explicit NodePool(
            std::pmr::memory_resource* resource,
            std::size_t chunkCapacity = 4096
        )
            : resource_(resource),
              chunkCapacity_(std::max<std::size_t>(1, chunkCapacity)),
              freeList_(resource)
        {
        }

        void Reserve(std::size_t nodeCapacity) {
            const std::size_t requiredChunks =
                nodeCapacity / chunkCapacity_ + (nodeCapacity % chunkCapacity_ != 0);

            while (chunks_.size() < requiredChunks) {
                chunks_.emplace_back(resource_, chunkCapacity_);
            }
        }

        [[nodiscard]] NodeId Allocate(
            State state,
            PathId path,
            std::optional<Cost> cost,
            NodeStatus status = NodeStatus::open
        ) {
            const bool recycled = !freeList_.empty();
            const NodeId id = recycled ? freeList_.back()
                                      : NodeId{static_cast<std::uint64_t>(nextUnused_)};
            if (!recycled) {
                if (!id.valid()) {
                    throw std::length_error("node handle capacity exhausted");
                }
                ensureChunkFor(id);
            }

            Chunk& chunk = chunks_.at(chunkIndex(id));
            const std::size_t offset = offsetOf(id);
            try {
                chunk.states[offset].emplace(std::move(state));
                chunk.costs[offset] = std::move(cost);
            } catch (...) {
                chunk.states[offset].reset();
                chunk.costs[offset].reset();
                throw;
            }
            chunk.paths[offset] = path;
            chunk.statuses[offset] = status;
            chunk.occupied[offset] = true;
            if (recycled) {
                freeList_.pop_back();
            } else {
                ++nextUnused_;
            }

            return id;
        }

        void Release(NodeId id) {
            Chunk& chunk = mutableChunkFor(id);
            const std::size_t offset = offsetOf(id);

            if (!chunk.occupied[offset]) {
                return;
            }

            // Grow the free list before destroying the node, so allocation
            // failure leaves a live, usable node instead of losing its slot.
            freeList_.push_back(id);
            chunk.states[offset].reset();
            chunk.costs[offset].reset();
            chunk.paths[offset] = PathId{};
            chunk.statuses[offset] = NodeStatus::open;
            chunk.occupied[offset] = false;
        }

        [[nodiscard]] State& StateAt(NodeId id) {
            RequireOccupied(id);
            return *mutableChunkFor(id).states[offsetOf(id)];
        }

        [[nodiscard]] const State& StateAt(NodeId id) const {
            RequireOccupied(id);
            return *chunkFor(id).states[offsetOf(id)];
        }

        [[nodiscard]] PathId PathAt(NodeId id) const {
            RequireOccupied(id);
            return chunkFor(id).paths[offsetOf(id)];
        }

        [[nodiscard]] std::optional<Cost>& CostAt(NodeId id) {
            RequireOccupied(id);
            return mutableChunkFor(id).costs[offsetOf(id)];
        }

        [[nodiscard]] const std::optional<Cost>& CostAt(NodeId id) const {
            RequireOccupied(id);
            return chunkFor(id).costs[offsetOf(id)];
        }

        [[nodiscard]] NodeStatus& StatusAt(NodeId id) {
            RequireOccupied(id);
            return mutableChunkFor(id).statuses[offsetOf(id)];
        }

        [[nodiscard]] std::size_t capacity() const noexcept {
            return chunks_.size() * chunkCapacity_;
        }

    private:
        struct Chunk {
            std::pmr::vector<std::optional<State>> states;
            std::pmr::vector<PathId> paths;
            std::pmr::vector<std::optional<Cost>> costs;
            std::pmr::vector<NodeStatus> statuses;
            std::pmr::vector<bool> occupied;

            Chunk(
                std::pmr::memory_resource* resource,
                std::size_t capacity
            )
                : states(resource),
                  paths(resource),
                  costs(resource),
                  statuses(resource),
                  occupied(resource)
            {
                // resize() creates the reusable slots once. State itself is not
                // constructed because each slot is std::optional<State>.
                states.resize(capacity);
                paths.resize(capacity);
                costs.resize(capacity);
                statuses.resize(capacity, NodeStatus::open);
                occupied.resize(capacity, false);
            }
        };

        std::pmr::memory_resource* resource_;
        std::size_t chunkCapacity_;
        std::vector<Chunk> chunks_;
        std::pmr::vector<NodeId> freeList_;
        std::size_t nextUnused_ = 0;

        void RequireOccupied(NodeId id) const {
            if (!chunkFor(id).occupied[offsetOf(id)]) {
                throw std::out_of_range("released NodeId");
            }
        }

        [[nodiscard]] std::size_t chunkIndex(NodeId id) const {
            return static_cast<std::size_t>(id.value / chunkCapacity_);
        }

        [[nodiscard]] std::size_t offsetOf(NodeId id) const {
            return static_cast<std::size_t>(id.value % chunkCapacity_);
        }

        void ensureChunkFor(NodeId id) {
            while (chunks_.size() <= chunkIndex(id)) {
                chunks_.emplace_back(resource_, chunkCapacity_);
            }
        }

        [[nodiscard]] Chunk& mutableChunkFor(NodeId id) {
            if (!id.valid() || id.value >= nextUnused_) {
                throw std::out_of_range("invalid NodeId");
            }
            return chunks_.at(chunkIndex(id));
        }

        [[nodiscard]] const Chunk& chunkFor(NodeId id) const {
            if (!id.valid() || id.value >= nextUnused_) {
                throw std::out_of_range("invalid NodeId");
            }
            return chunks_.at(chunkIndex(id));
        }
    };

    // One monotonic PMR resource owns all internal vector backing allocations.
    // Individual node slots are recycled by NodePool; bulk allocation metadata
    // is reclaimed in one operation when SearchMemory is destroyed.
    class SearchMemory {
    public:
        explicit SearchMemory(
            std::size_t initialBytes = 1024 * 1024
        )
            : initialBuffer_(initialBytes),
              resource_(
                  initialBuffer_.data(),
                  initialBuffer_.size(),
                  std::pmr::new_delete_resource()
              )
        {
        }

        [[nodiscard]] std::pmr::memory_resource* resource() noexcept {
            return &resource_;
        }

    private:
        std::vector<std::byte> initialBuffer_;
        std::pmr::monotonic_buffer_resource resource_;
    };
}
