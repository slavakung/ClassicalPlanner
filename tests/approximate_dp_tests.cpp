#include "ApproximateDPPlanner.h"
#include "GraphProblem.h"
#include "GridProblem.h"
#include "TestSupport.h"
#include "pddl/Adapter.h"

#include <iostream>
#include <random>

namespace {
using namespace planning;
using test::Require;
using test::Throws;
using DP = ApproximateDPPlanner<int,int>;

template<class Problem, class PlanType>
void Replay(const Problem& problem, const PlanType& plan) {
    bool valid = false;
    problem.ForEachInitialState([&](const auto& initial) {
        auto state = initial;
        float cost = 0;
        for (const auto& action : plan.actions) {
            const auto next = problem.Apply(state, action);
            if (!next) return true;
            cost += problem.StepCost(state, action, *next).value_or(0);
            state = *next;
        }
        cost += problem.TerminalCost(state).value_or(0);
        valid = problem.IsGoal(state) && (!plan.totalCost || *plan.totalCost == cost);
        return !valid;
    });
    Require(valid && plan.status == PlanStatus::success && plan.stageLen == plan.actions.size(),
            "DP result independently replays with its reported cost");
}

void RolloutImprovement() {
    test::GraphProblem graph;
    // A one-step c+h base policy with h=0 takes the first edge (cost 1) and
    // subsequently pays 100. Two-step rollout observes a complete cost-5 route.
    graph.edges = {{0,1,1},{0,2,3},{1,4,100},{2,4,2}};
    ApproximateDPOptions options;
    options.mode = ApproximateDPMode::rollout;
    options.rolloutDepth = 2;
    options.trialDepth = 1;
    options.maxDepth = 2;
    DP planner(options);
    const auto plans = planner.Search(graph);
    Require(plans.size() == 1 && plans[0].actions.front() == 1 && plans[0].totalCost == 5,
            "rollout improves the greedy base trajectory from 101 to 5");
    Replay(graph, plans[0]);
    Require(planner.Statistics().committedDecisions == 1 && planner.Statistics().rolloutEvaluations == 2,
            "one committed decision retains a complete evaluated suffix beyond that decision budget");
    options.rolloutDepth = 1;
    Require(DP(options).Search(graph).empty(), "truncated evaluation with no complete incumbent returns no plan");
    options.rolloutDepth = 2;
    options.maxDepth = 1;
    Require(DP(options).Search(graph).empty(), "rollout suffix obeys the total depth bound");
    options.maxPlans = 0;
    Require(DP(options).Search(graph).empty(), "zero plan-output budget");
}

void LearningAndBudgets() {
    test::GraphProblem graph;
    graph.edges = {{0,1,1},{0,2,2},{2,4,1}};
    ApproximateDPOptions options;
    options.trials = 1;
    options.trialDepth = 4;
    options.maxDepth = 4;
    DP first(options);
    Require(first.Search(graph).empty(), "first RTDP trial follows a misleading dead end");
    Require(first.LearnedValue(graph,1) && std::isinf(*first.LearnedValue(graph,1)),
            "a successor-free non-goal receives an infinite value");
    options.trials = 2;
    DP learned(options);
    const auto plans = learned.Search(graph);
    Require(plans.size() == 1 && plans[0].totalCost == 3, "later trial improves its policy using Bellman learning");
    Replay(graph, plans[0]);
    Require(learned.Statistics().trialsStarted == 2, "RTDP trial budget is explicit");
    Require(*learned.LearnedValue(graph,0) <= 3 && *learned.LearnedValue(graph,0) > 2.9L,
            "reverse sweep propagates the successful lower bound");

    graph.edges = {{0,0,0},{0,4,1}};
    options.prunePathCycles = false;
    options.trialDepth = 3;
    DP cyclic(options);
    Require(cyclic.Search(graph).empty(), "zero-cost greedy self-loop can exhaust an approximate trial");
    Require(cyclic.Statistics().committedDecisions == 6 && *cyclic.LearnedValue(graph,0) == 0,
            "zero-cost cycles stop at the algorithmic budget without a false dead-end proof");
    options.prunePathCycles = true;
    DP pruned(options);
    const auto escaped = pruned.Search(graph);
    Require(escaped.size() == 1 && escaped[0].totalCost == 1, "trajectory cycle pruning can take the exit");
    Require(*pruned.LearnedValue(graph,0) == 0, "Bellman backup still includes the trajectory-forbidden self-loop");
    Replay(graph, escaped[0]);

    graph.edges = {{0,1,1},{1,4,1}};
    options.trials = 1;
    options.trialDepth = 1;
    DP truncated(options);
    Require(truncated.Search(graph).empty(), "trial depth is distinct from a feasibility proof");
    Require(truncated.LearnedValue(graph,1) && *truncated.LearnedValue(graph,1) == 0,
            "budget leaf retains its initialization rather than infinity");
    options.mode = ApproximateDPMode::rtdpAStar;
    const auto completed = DP(options).Search(graph);
    Require(completed.size() == 1 && completed[0].totalCost == 2, "A* continues beyond the warm-up trial budget");
    Replay(graph, completed[0]);
}

void PartitionsAndTerminalCosts() {
    test::GraphProblem graph;
    graph.edges = {{0,4,1},{0,1,5},{1,4,5}};
    ApproximateDPOptions options;
    options.mode = ApproximateDPMode::rtdpAStar;
    options.trials = 2;
    options.maxDepth = 3;
    options.trialDepth = 3;
    DP restricted(options);
    std::vector<Plan<int,float>> plans;
    restricted.Search(graph, [&](const auto& p) { plans.push_back(p); return true; }, {1,2});
    Require(plans.size() == 1 && plans[0].totalCost == 10 && plans[0].actions[0] == 1,
            "ADP-A* preserves root ownership in the final search");
    Require(*restricted.LearnedValue(graph,0) <= 1,
            "root-restricted training backs up the unowned cheap action too");
    Replay(graph, plans[0]);

    graph.edges = {{0,1,1},{0,4,1},{1,0,0}};
    options.prunePathCycles = false;
    DP returning(options);
    plans.clear();
    returning.Search(graph, [&](const auto& p) { plans.push_back(p); return true; }, {0,2});
    Require(plans.size() == 1 && plans[0].totalCost == 2 && plans[0].stageLen == 3,
            "returning to an initial state does not reapply root-only restrictions");
    Replay(graph, plans[0]);

    class GoalFees : public test::GraphProblem {
    public:
        std::optional<float> TerminalCost(const int& state) const override {
            return state == 2 ? 100.0F : 0.0F;
        }
    } fees;
    fees.goals = {2,4};
    fees.edges = {{0,2,1},{2,4,1},{0,4,5}};
    options.prunePathCycles = true;
    options.trials = 3;
    DP exact(options);
    const auto cheap = exact.Search(fees);
    Require(cheap.size() == 1 && cheap[0].totalCost == 2,
            "A* can continue through an expensive terminal goal to a cheaper goal");
    Require(*exact.LearnedValue(fees,2) == 0, "learned goal step value excludes terminal charges");
    Replay(fees, cheap[0]);
    options.mode = ApproximateDPMode::rollout;
    options.rolloutDepth = 3;
    const auto approximate = DP(options).Search(fees);
    Require(approximate.size() == 1 && approximate[0].totalCost == 5,
            "first-goal rollout is honestly approximate with unequal terminal charges");
    Replay(fees, approximate[0]);

    for (auto mode : {ApproximateDPMode::rollout, ApproximateDPMode::rtdp, ApproximateDPMode::rtdpAStar}) {
        options.mode = mode;
        options.trials = 0;
        options.maxDepth = 0;
        graph.initial = {4};
        std::size_t count = 0;
        for (std::size_t rank = 0; rank < 3; ++rank)
            DP(options).Search(graph, [&](const auto& p) { Replay(graph,p); ++count; return true; }, {rank,3});
        Require(count == 1, "initial goal belongs only to partition zero, even with zero warm-up trials");
    }
}

void AdmissibilityOracle() {
    std::mt19937 random(481703);
    for (unsigned repeat = 0; repeat < 35; ++repeat) {
        test::GraphProblem graph;
        graph.goals = {5};
        graph.edges.push_back({0,5,9});
        for (int from = 0; from < 6; ++from)
            for (int to = 0; to < 6; ++to)
                if (random() % 4 == 0) graph.edges.push_back({from,to,static_cast<float>(random()%6)});
        std::vector<float> optimum(6, std::numeric_limits<float>::infinity());
        optimum[5] = 0;
        for (int iteration = 0; iteration < 6; ++iteration)
            for (const auto& edge : graph.edges)
                optimum[edge.from] = std::min(optimum[edge.from], edge.cost + optimum[edge.to]);
        ApproximateDPOptions options;
        options.mode = ApproximateDPMode::rtdpAStar;
        options.trials = 4;
        options.trialDepth = 3;
        options.maxDepth = 5;
        options.admissibleInitialization = true;
        const auto h = [&](int state) { return std::isfinite(optimum[state]) ? optimum[state] * 0.25F : 0.0F; };
        DP planner(options,h);
        const auto result = planner.Search(graph);
        Require(result.size() == 1 && result[0].totalCost == optimum[0],
                "ADP-A* agrees with independent exact shortest-path costs");
        Replay(graph,result[0]);
        for (int state = 0; state < 6; ++state)
            if (auto learned = planner.LearnedValue(graph,state))
                Require(*learned <= static_cast<long double>(optimum[state]),
                        "every observed learned value is an admissible unrestricted lower bound");
        Require(planner.Statistics().learnedStates <= 6,
                "constant-hash history keys remain distinct through exact equality checks");
    }
}

void CostContractsAndErrors() {
    test::GraphProblem graph;
    graph.edges = {{0,1,7},{1,4,9}};
    graph.hasCost = false;
    ApproximateDPOptions options;
    options.maxDepth = 2;
    options.rolloutDepth = 2;
    options.trialDepth = 2;
    for (auto mode : {ApproximateDPMode::rollout, ApproximateDPMode::rtdp, ApproximateDPMode::rtdpAStar}) {
        options.mode = mode;
        const auto result = DP(options).Search(graph);
        Require(result.size() == 1 && !result[0].totalCost, "costless results preserve absent domain costs");
        Replay(graph,result[0]);
    }
    class TerminalOnly : public test::GraphProblem {
    public:
        bool HasStepCost() const override { return false; }
        std::optional<float> StepCost(const int&,const int&,const int&) const override { return std::nullopt; }
    } terminalOnly;
    terminalOnly.edges = graph.edges;
    terminalOnly.terminal = 7;
    for (auto mode : {ApproximateDPMode::rollout, ApproximateDPMode::rtdp, ApproximateDPMode::rtdpAStar}) {
        options.mode = mode;
        const auto result = DP(options).Search(terminalOnly);
        Require(result.size() == 1 && result[0].totalCost == 7, "terminal-only models use zero missing step charges");
        Replay(terminalOnly,result[0]);
    }
    options.mode = ApproximateDPMode::rtdpAStar;
    Throws([&] { DP bad(options, [](int) { return 0.0F; }); }, "supplied A* initialization needs an admissibility contract");
    options.admissibleInitialization = true;
    options.mode = ApproximateDPMode::rtdp;
    for (const auto bad : {-1.0F, std::numeric_limits<float>::quiet_NaN()})
        Throws([&] { (void)DP(options,[=](int) { return bad; }).Search(graph); }, "invalid DP heuristic rejected");
    graph.hasCost = true;
    graph.edges[0].cost = -1;
    Throws([&] { (void)DP(options).Search(graph); }, "negative transition costs rejected");
    graph.edges[0].cost = std::numeric_limits<float>::infinity();
    Throws([&] { (void)DP(options).Search(graph); }, "infinite transition costs rejected");
    options.rolloutDepth = 0;
    Throws([&] { DP invalid(options); }, "zero rollout lookahead rejected");
}

void FloatingPointRegime() {
    ApproximateDPOptions options;
    options.mode = ApproximateDPMode::rtdpAStar;
    options.trials = 0; // final A* must validate costs independently of warm-up
    options.maxDepth = 10;
    test::GraphProblem graph;
    graph.edges = {{0,1,0.5F},{1,4,0.5F}};
    Throws([&] { (void)DP(options).Search(graph); }, "ADP-A* rejects fractional costs even with zero warm-up trials");
    for (auto mode : {ApproximateDPMode::rollout, ApproximateDPMode::rtdp}) {
        auto approximate = options;
        approximate.mode = mode;
        approximate.trials = 2;
        const auto plans = DP(approximate).Search(graph);
        Require(plans.size() == 1 && plans[0].totalCost == 1,
                "approximate rollout and RTDP retain fractional-cost support");
        Replay(graph,plans[0]);
    }
    graph.goals = {6};
    graph.edges = {{0,6,100000008.0F},{0,1,0},{1,2,100000000.0F},
                   {2,3,4},{3,4,4},{4,5,4},{5,6,4}};
    float forward = 100000000.0F;
    for (int i = 0; i < 4; ++i) forward += 4.0F;
    Require(forward == 100000000.0F, "regression reproduces nonassociative float path accumulation");
    Throws([&] { (void)DP(options).Search(graph); }, "ADP-A* rejects large floating costs that invalidate Bellman reassociation");
    const auto boundary = std::ldexp(1.0F,std::numeric_limits<float>::digits);
    graph.goals = {4};
    graph.edges = {{0,1,boundary - 1},{1,4,2}};
    Throws([&] { (void)DP(options).Search(graph); }, "rounded total at the exact-integer boundary is rejected before delivery");
    graph.edges = {{0,1,boundary - 2},{1,4,1}};
    const auto exact = DP(options).Search(graph);
    Require(exact.size() == 1 && exact[0].totalCost == boundary - 1,
            "integer accumulation strictly below the exact range boundary is supported");
    Replay(graph,exact[0]);
    graph.edges = {{0,4,1}};
    graph.terminal = 0.5F;
    Throws([&] { (void)DP(options).Search(graph); }, "final A* validates fractional terminal fees too");


}

void CustomHistoryEquality() {
    class AliasedKeys : public test::GraphProblem {
    public:
        int MakeHistoryKey(const int& state) const override { return 2 * state + static_cast<int>(calls_++ % 2); }
        bool HistoryKeysEqual(const int& left,const int& right) const override { return left / 2 == right / 2; }
        std::size_t HistoryKeyHash(const int& key) const override { return static_cast<std::size_t>((key / 2) % 2); }
    private:
        mutable unsigned calls_ = 0;
    } graph;
    graph.edges = {{0,0,0},{0,1,1},{1,4,1}};
    for (auto mode : {ApproximateDPMode::rollout, ApproximateDPMode::rtdp, ApproximateDPMode::rtdpAStar}) {
        ApproximateDPOptions options;
        options.mode = mode;
        options.maxDepth = 2;
        options.trials = 2;
        options.trialDepth = 2;
        options.rolloutDepth = 2;
        DP planner(options);
        const auto result = planner.Search(graph);
        Require(result.size() == 1 && result[0].totalCost == 2,
                "custom history equality prunes a zero-cost self-loop despite unequal key representations");
        Require(planner.Statistics().learnedStates == 3,
                "value table uses caller equality inside hash buckets");
        Replay(graph,result[0]);
    }
}

void NativeDomains() {
    GridProblem grid(3,2,{0,0},{2,1});
    const auto logic = pddl::LoadGroundedProblemFromText(
        "(define (domain d) (:predicates (p) (q)) "
        "(:action begin :precondition (and) :effect (p)) "
        "(:action finish :precondition (p) :effect (q)))",
        "(define (problem x) (:domain d) (:init) (:goal (q)))");
    for (auto mode : {ApproximateDPMode::rollout, ApproximateDPMode::rtdp, ApproximateDPMode::rtdpAStar}) {
        ApproximateDPOptions options;
        options.mode = mode;
        options.maxDepth = 5;
        options.trials = 4;
        options.trialDepth = 5;
        options.rolloutDepth = 5;
        std::size_t count = 0;
        test::Run([=] { return ApproximateDPPlanner<GridMove,GridState>(options); }, grid,
            [&](const auto& plan) { Replay(grid,plan); ++count; return true; });
        Require(count == 1, "DP works through the native grid problem");
        count = 0;
        test::Run([=] { return ApproximateDPPlanner<pddl::PDDLAction,DynamicBitset,float,LogicStateKey>(options); },
            logic, [&](const auto& plan) { Replay(logic,plan); ++count; return true; });
        Require(count == 1, "DP works through native propositional expansion");
    }
}
}

int main() {
    try {
        RolloutImprovement();
        LearningAndBudgets();
        PartitionsAndTerminalCosts();
        AdmissibilityOracle();
        CostContractsAndErrors();
        FloatingPointRegime();
        CustomHistoryEquality();
        NativeDomains();
        std::cout << "Approximate DP: " << test::checks << " checks passed\n";
        return 0;
    } catch (const std::exception& error) {
        std::cerr << error.what() << '\n';
        return 1;
    }
}
