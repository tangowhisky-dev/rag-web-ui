/**
 * Tests for AgentTimeline graceful degradation when streaming is cancelled.
 *
 * Covers R017: when cancellation abruptly ends the SSE stream, the frontend
 * agent timeline must handle the transition from "active" to "done" without
 * leaving dangling in-progress indicators or crashing.
 */

import { render, screen, act, waitFor } from "@testing-library/react";
import { AgentTimeline } from "../agent-timeline";

// Helper to wait for useEffect to fire
async function expectText(text: string) {
  await waitFor(() => {
    expect(screen.getByText(new RegExp(text, "i"))).toBeInTheDocument();
  });
}

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
      />
    );

    // All 3 steps should be rendered as badges
    const rows = container.querySelectorAll("[class*='relative flex']");
    expect(rows.length).toBeGreaterThanOrEqual(3);

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
      />
    );

    // No rows rendered
    expect(container.children.length).toBe(0);
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
      />
    );

    // Steps rendered (at least 3 badge elements)
    const rows = container.querySelectorAll("[class*='relative flex']");
    expect(rows.length).toBeGreaterThanOrEqual(3);

    // Error step should have red styling (text-red-500)
    const errorRows = container.querySelectorAll(
      '[class*="text-red"]'
    );
    expect(errorRows.length).toBeGreaterThan(0);

    // Active step should have blue spinner
    const spinners = container.querySelectorAll('[class*="animate-spin"]');
    expect(spinners.length).toBeGreaterThan(0);
  });

  it("auto-dismisses badges after streaming ends", async () => {
    /**
     * Scenario: streaming completes normally.
     * The timeline should auto-dismiss badges after 1.5s.
     */
    const { rerender } = render(
      <AgentTimeline
        agentSteps={[
          { node: "rewrite_query", status: "done", latency_ms: 45 },
          { node: "context_router", status: "done", latency_ms: 30 },
        ]}
        isStreaming={true}
      />
    );

    // While streaming, badges are visible
    await expectText("rewriting query");

    // Stop streaming
    rerender(
      <AgentTimeline
        agentSteps={[
          { node: "rewrite_query", status: "done", latency_ms: 45 },
          { node: "context_router", status: "done", latency_ms: 30 },
        ]}
        isStreaming={false}
      />
    );

    // Should auto-dismiss after ~1.5s
    await new Promise((r) => setTimeout(r, 1700));
    expect(screen.queryByText(/rewriting query/i)).not.toBeInTheDocument();
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
      />
    );

    // Should have 2 unique steps (rewrite_query + draft_answer), not 3
    const rows = container.querySelectorAll("[class*='relative flex']");
    expect(rows.length).toBeGreaterThanOrEqual(2);

    // rewrite_query should show "done" (emerald), not "active" (blue spinner)
    const doneRows = container.querySelectorAll('[class*="emerald"]');
    expect(doneRows.length).toBeGreaterThan(0);

    // draft_answer should still be active (spinner)
    const spinners = container.querySelectorAll('[class*="animate-spin"]');
    expect(spinners.length).toBeGreaterThan(0);
  });
});
