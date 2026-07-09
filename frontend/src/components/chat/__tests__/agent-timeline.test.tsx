import React from "react";
import { render, screen, act, waitFor } from "@testing-library/react";
import "@testing-library/jest-dom";
import { AgentTimeline } from "../agent-timeline";

// Helper to wait for useEffect to fire
async function expectText(text: string) {
  await waitFor(() => {
    expect(screen.getByText(new RegExp(text, "i"))).toBeInTheDocument();
  });
}

describe("AgentTimeline (transient)", () => {
  describe("initial render", () => {
    it("renders nothing when no agentSteps", () => {
      const { container } = render(<AgentTimeline isStreaming={true} />);
      expect(container.innerHTML).toBe("");
    });

    it("renders nothing when empty agentSteps", () => {
      const { container } = render(
        <AgentTimeline agentSteps={[]} isStreaming={true} />
      );
      expect(container.innerHTML).toBe("");
    });
  });

  describe("active step during streaming", () => {
    it("renders active step badge", async () => {
      render(
        <AgentTimeline
          agentSteps={[
            { node: "rewrite_query", latency_ms: 50, status: "active" },
          ]}
          isStreaming={true}
        />
      );
      await expectText("rewriting query");
    });

    it("renders multiple active steps", async () => {
      render(
        <AgentTimeline
          agentSteps={[
            { node: "rewrite_query", latency_ms: 50, status: "active" },
            { node: "kb_retrieval", latency_ms: 200, status: "active" },
          ]}
          isStreaming={true}
        />
      );
      await expectText("rewriting query");
      await expectText("retrieving");
    });

    it("renders done step", async () => {
      render(
        <AgentTimeline
          agentSteps={[
            { node: "rewrite_query", latency_ms: 50, status: "done" },
          ]}
          isStreaming={true}
        />
      );
      await expectText("rewriting query");
    });

    it("renders error step", async () => {
      render(
        <AgentTimeline
          agentSteps={[
            { node: "rewrite_query", latency_ms: 50, status: "error" },
          ]}
          isStreaming={true}
        />
      );
      await expectText("rewriting query");
    });
  });

  describe("step deduplication", () => {
    it("keeps latest status per node", async () => {
      render(
        <AgentTimeline
          agentSteps={[
            { node: "rewrite_query", latency_ms: 50, status: "active" },
            { node: "rewrite_query", latency_ms: 60, status: "done" },
          ]}
          isStreaming={true}
        />
      );
      await expectText("rewriting query");
      const elements = screen.queryAllByText(/rewriting query/i);
      expect(elements).toHaveLength(1);
    });

    it("upgrades active to done", async () => {
      render(
        <AgentTimeline
          agentSteps={[
            { node: "kb_retrieval", latency_ms: 100, status: "active" },
            { node: "kb_retrieval", latency_ms: 150, status: "done" },
          ]}
          isStreaming={true}
        />
      );
      // When done, the label "Retrieving context…" is shown
      await expectText("retrieving context");
      const elements = screen.queryAllByText(/retrieving context/i);
      expect(elements).toHaveLength(1);
    });
  });

  describe("auto-dismiss after streaming ends", () => {
    it("disables after streaming completes", async () => {
      jest.useFakeTimers();

      const { rerender } = render(
        <AgentTimeline
          agentSteps={[
            { node: "rewrite_query", latency_ms: 50, status: "done" },
            { node: "kb_retrieval", latency_ms: 200, status: "done" },
          ]}
          isStreaming={true}
        />
      );

      // While streaming, badges are visible
      await expectText("rewriting query");

      // Stop streaming
      await act(async () => {
        rerender(
          <AgentTimeline
            agentSteps={[
              { node: "rewrite_query", latency_ms: 50, status: "done" },
              { node: "kb_retrieval", latency_ms: 200, status: "done" },
            ]}
            isStreaming={false}
          />
        );
      });

      // Advance past 1.5s (auto-dismiss threshold)
      await act(async () => {
        jest.advanceTimersByTime(1600);
      });

      expect(await screen.queryByText(/rewriting query/i)).not.toBeInTheDocument();
      jest.useRealTimers();
    });
  });

  describe("step ordering", () => {
    it("renders steps in order of first appearance", async () => {
      const { container } = render(
        <AgentTimeline
          agentSteps={[
            { node: "rewrite_query", latency_ms: 50, status: "active" },
            { node: "kb_retrieval", latency_ms: 200, status: "active" },
            { node: "generate_answer", latency_ms: 0, status: "active" },
          ]}
          isStreaming={true}
        />
      );

      await expectText("rewriting query");
      const allBadges = Array.from(container.querySelectorAll("[class*='relative flex']"));
      expect(allBadges.length).toBe(3);
    });
  });

  describe("icon map", () => {
    it("renders correct icons for known nodes", async () => {
      render(
        <AgentTimeline
          agentSteps={[
            { node: "rewrite_query", latency_ms: 50, status: "active" },
            { node: "kb_retrieval", latency_ms: 200, status: "done" },
            { node: "generate_answer", latency_ms: 0, status: "done" },
          ]}
          isStreaming={false}
        />
      );

      // Active step shows activeLabel "Rewriting query…"
      await expectText("rewriting query");
      // Done steps show label "Retrieving context…" and "Generating answer…"
      await expectText("retrieving context");
      await expectText("generating answer");
    });

    it("handles unknown nodes with fallback icon", async () => {
      render(
        <AgentTimeline
          agentSteps={[
            { node: "custom_node", latency_ms: 100, status: "done" },
          ]}
          isStreaming={false}
        />
      );

      // Unknown node shows its node name as fallback
      await expectText("custom_node");
    });
  });
});
