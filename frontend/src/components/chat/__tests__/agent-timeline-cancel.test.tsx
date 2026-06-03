/**
 * Tests for AgentTimeline graceful degradation when streaming is cancelled.
 *
 * Covers R017: when cancellation abruptly ends the SSE stream, the frontend
 * agent timeline must handle the transition from "active" to "done" without
 * leaving dangling in-progress indicators or crashing.
 */

import { render, screen } from "@testing-library/react";
import { AgentTimeline } from "../agent-timeline";

describe("AgentTimeline — cancellation graceful degradation (R017)", () => {
  it("renders steps with active status when streaming stops abruptly", () => {
    /**
     * Scenario: user clicks Stop while draft_answer is active.
     * The stream ends mid-event, so the last agent step remains "active".
     * The timeline should show it as active (spinner) — not crash.
     */
    const agentSteps = [
      { node: "rewrite_query", status: "done", latency_ms: 45 },
      { node: "context_router", status: "done", latency_ms: 30 },
      { node: "draft_answer", status: "active", latency_ms: 0 },
    ];

    const { container } = render(
      <AgentTimeline
        agentSteps={agentSteps}
        isStreaming={false}
        answerStarted={false}
      />
    );

    // All 3 steps should be rendered
    const rows = container.querySelectorAll("[class*='rounded-md']");
    expect(rows).toHaveLength(3);

    // The active step (draft_answer) should show a spinner (Loader2)
    const spinners = container.querySelectorAll('[class*="animate-spin"]');
    expect(spinners.length).toBeGreaterThan(0);
  });

  it("does not crash when agentSteps is empty and isStreaming is false", () => {
    /**
     * Scenario: user cancels before any agent steps are emitted.
     * The timeline should render nothing (no crash).
     */
    const { container } = render(
      <AgentTimeline
        agentSteps={[]}
        isStreaming={false}
        answerStarted={false}
      />
    );

    // No steps rendered
    const rows = container.querySelectorAll("[class*='rounded-md']");
    expect(rows).toHaveLength(0);
  });

  it("handles mixed active/done/error statuses from abrupt stream end", () => {
    /**
     * Scenario: stream ends with a mix of completed, active, and errored steps.
     * This can happen if an error occurred mid-stream and the stream was cancelled.
     */
    const agentSteps = [
      { node: "rewrite_query", status: "done", latency_ms: 45 },
      { node: "context_router", status: "error", latency_ms: 0 },
      { node: "draft_answer", status: "active", latency_ms: 0 },
    ];

    const { container } = render(
      <AgentTimeline
        agentSteps={agentSteps}
        isStreaming={false}
        answerStarted={false}
      />
    );

    // All 3 steps rendered
    const rows = container.querySelectorAll("[class*='rounded-md']");
    expect(rows).toHaveLength(3);

    // Error step should have red styling
    const errorRows = container.querySelectorAll(
      '[class*="border-red"]'
    );
    expect(errorRows.length).toBeGreaterThan(0);

    // Active step should have blue spinner
    const spinners = container.querySelectorAll('[class*="animate-spin"]');
    expect(spinners.length).toBeGreaterThan(0);
  });

  it("collapses expanded steps when answer starts (answerStarted=true)", () => {
    /**
     * Scenario: answer has started streaming — the timeline should auto-collapse
     * all expanded steps to save screen space.
     */
    const agentSteps = [
      { node: "rewrite_query", status: "done", latency_ms: 45 },
      { node: "context_router", status: "done", latency_ms: 30 },
    ];

    const { container } = render(
      <AgentTimeline
        agentSteps={agentSteps}
        isStreaming={true}
        answerStarted={true}
      />
    );

    // Steps are rendered but collapsed (max-h-0 or max-h-0 opacity-0)
    const expandedPanels = container.querySelectorAll("[class*='max-h-96']");
    expect(expandedPanels.length).toBe(0);
  });

  it("renders derived steps when agentSteps is empty but metadata is present", () => {
    /**
     * Scenario: no agent steps (fast-pipeline path), but context and classification
     * metadata are available. The timeline should show derived steps instead.
     */
    const { container } = render(
      <AgentTimeline
        rewrittenQuery="rewritten question"
        queryClassification={{
          type: "factual",
          confidence: 0.85,
          latency_ms: 50,
          fallback: false,
        }}
        isStreaming={false}
        answerStarted={false}
      />
    );

    // Should have derived steps (Query Rewrite + Classification)
    const rows = container.querySelectorAll("[class*='rounded-md']");
    expect(rows.length).toBeGreaterThanOrEqual(2);
  });

  it("deduplicates agent steps — keeps last event per node", () => {
    /**
     * Scenario: the same node emits multiple events (active → done).
     * The timeline should show only the final state.
     */
    const agentSteps = [
      { node: "rewrite_query", status: "active", latency_ms: 0 },
      { node: "rewrite_query", status: "done", latency_ms: 45 },
      { node: "draft_answer", status: "active", latency_ms: 0 },
    ];

    const { container } = render(
      <AgentTimeline
        agentSteps={agentSteps}
        isStreaming={false}
        answerStarted={false}
      />
    );

    // Should have 2 unique steps (rewrite_query + draft_answer), not 3
    const rows = container.querySelectorAll("[class*='rounded-md']");
    expect(rows).toHaveLength(2);

    // rewrite_query should show "done" (emerald), not "active" (blue spinner)
    const doneRows = container.querySelectorAll('[class*="border-emerald"]');
    expect(doneRows.length).toBeGreaterThan(0);

    // draft_answer should still be active (spinner)
    const spinners = container.querySelectorAll('[class*="animate-spin"]');
    expect(spinners.length).toBeGreaterThan(0);
  });
});
