#pragma once

#include <algorithm>
#include <bit>
#include <cstddef>
#include <cstdint>
#include <optional>
#include <string>
#include <unordered_map>
#include <utility>
#include <vector>

namespace planning
{
    // Interning replaces repeated logical strings in every state/action with
    // compact integer IDs. Human-readable strings remain available for display.
    class SymbolTable {
    public:
        using SymbolId = std::uint32_t;

        SymbolId Intern(std::string symbol) {
            const auto existing = ids_.find(symbol);
            if (existing != ids_.end()) {
                return existing->second;
            }

            const SymbolId id =
                static_cast<SymbolId>(symbols_.size());
            ids_.emplace(symbol, id);
            symbols_.push_back(std::move(symbol));
            return id;
        }

        [[nodiscard]] const std::string& Name(SymbolId id) const {
            return symbols_.at(id);
        }

        [[nodiscard]] std::optional<SymbolId> Find(
            const std::string& symbol
        ) const {
            const auto it = ids_.find(symbol);
            if (it == ids_.end()) {
                return std::nullopt;
            }
            return it->second;
        }

        [[nodiscard]] std::size_t Size() const noexcept {
            return symbols_.size();
        }

    private:
        std::unordered_map<std::string, SymbolId> ids_;
        std::vector<std::string> symbols_;
    };

    // Runtime-sized bitset. STRIPS-style precondition/effect operations become
    // word-level integer operations instead of tree/set operations on strings.
    class DynamicBitset {
    public:
        DynamicBitset() = default;

        explicit DynamicBitset(std::size_t bitCount)
            : words_(bitCount / 64U + (bitCount % 64U != 0), 0)
        {
        }

        explicit DynamicBitset(std::vector<std::uint64_t> words)
            : words_(std::move(words))
        {
        }

        void Resize(std::size_t bitCount) {
            words_.resize(bitCount / 64U + (bitCount % 64U != 0), 0);
            if (!words_.empty() && bitCount % 64U != 0) {
                words_.back() &= (std::uint64_t{1} << (bitCount % 64U)) - 1U;
            }
        }

        void Set(std::size_t bit) {
            Ensure(bit);
            words_[bit / 64U] |=
                (std::uint64_t{1} << (bit % 64U));
        }

        void Reset(std::size_t bit) {
            if (bit / 64U >= words_.size()) {
                return;
            }
            words_[bit / 64U] &=
                ~(std::uint64_t{1} << (bit % 64U));
        }

        [[nodiscard]] bool Test(std::size_t bit) const {
            if (bit / 64U >= words_.size()) {
                return false;
            }
            return (words_[bit / 64U] &
                    (std::uint64_t{1} << (bit % 64U))) != 0;
        }

        [[nodiscard]] bool ContainsAll(
            const DynamicBitset& required
        ) const {
            for (std::size_t i = 0;
                 i < required.words_.size();
                 ++i) {
                const std::uint64_t have =
                    i < words_.size() ? words_[i] : 0;

                if ((have & required.words_[i]) !=
                    required.words_[i]) {
                    return false;
                }
            }
            return true;
        }

        [[nodiscard]] bool Intersects(const DynamicBitset& other) const noexcept {
            const std::size_t count =
                std::min(words_.size(), other.words_.size());

            for (std::size_t i = 0; i < count; ++i) {
                if ((words_[i] & other.words_[i]) != 0) {
                    return true;
                }
            }
            return false;
        }

        void UnionWith(const DynamicBitset& other) {
            if (words_.size() < other.words_.size()) {
                words_.resize(other.words_.size(), 0);
            }

            for (std::size_t i = 0;
                 i < other.words_.size();
                 ++i) {
                words_[i] |= other.words_[i];
            }
        }

        void DifferenceWith(const DynamicBitset& other) {
            const std::size_t count =
                std::min(words_.size(), other.words_.size());

            for (std::size_t i = 0; i < count; ++i) {
                words_[i] &= ~other.words_[i];
            }
        }

        [[nodiscard]] std::uint64_t Hash() const noexcept {
            // FNV-1a-style word hash. Hashing accelerates comparisons and future
            // visited tables; correctness below never depends on hash alone.
            std::uint64_t hash = 1469598103934665603ULL;
            // Equality ignores zero padding. Hash the same canonical range so
            // equal states also have equal keys after device-width padding.
            std::size_t count = words_.size();
            while (count != 0 && words_[count - 1] == 0) {
                --count;
            }
            for (std::size_t i = 0; i < count; ++i) {
                hash ^= words_[i];
                hash *= 1099511628211ULL;
            }
            return hash;
        }

        [[nodiscard]] std::uint32_t Population() const noexcept {
            std::uint32_t result = 0;
            for (const std::uint64_t word : words_) {
                result += static_cast<std::uint32_t>(
                    std::popcount(word)
                );
            }
            return result;
        }

        [[nodiscard]] bool operator==(
            const DynamicBitset& other
        ) const noexcept {
            const std::size_t maxSize =
                std::max(words_.size(), other.words_.size());

            for (std::size_t i = 0; i < maxSize; ++i) {
                const std::uint64_t left =
                    i < words_.size() ? words_[i] : 0;
                const std::uint64_t right =
                    i < other.words_.size() ? other.words_[i] : 0;
                if (left != right) {
                    return false;
                }
            }
            return true;
        }

        [[nodiscard]] const std::vector<std::uint64_t>& Words() const noexcept {
            return words_;
        }

    private:
        void Ensure(std::size_t bit) {
            const std::size_t requiredWords = bit / 64U + 1U;
            if (words_.size() < requiredWords) {
                words_.resize(requiredWords, 0);
            }
        }

        std::vector<std::uint64_t> words_;
    };

    // Exact compact history key for logic states. The previous revision stored
    // only hash+population, which had a theoretical collision correctness bug.
    // Here we keep the compressed 64-bit words as an exact fallback while the
    // hash/population provide fast inequality tests.
    struct LogicStateKey {
        std::uint64_t hash = 0;
        std::uint32_t population = 0;
        std::vector<std::uint64_t> words;

        [[nodiscard]] bool operator==(
            const LogicStateKey& other
        ) const noexcept {
            if (hash != other.hash ||
                population != other.population) {
                return false;
            }
            return words == other.words;
        }
    };

    inline LogicStateKey MakeLogicStateKey(
        const DynamicBitset& bits
    ) {
        auto canonicalWords = bits.Words();
        while (!canonicalWords.empty() && canonicalWords.back() == 0) {
            canonicalWords.pop_back();
        }
        return LogicStateKey{
            bits.Hash(),
            bits.Population(),
            std::move(canonicalWords)
        };
    }
}
