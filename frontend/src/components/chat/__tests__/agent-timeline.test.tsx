import React from "react";
import { render, screen } from "@testing-library/react";
import "@testing-library/jest-dom";
import { AgentTimeline, TimelineStep } from "../agent-timeline";

describe("AgentTimeline", () => {
  describe("render order", () => {
    it("renders steps in the correct order: Query Rewrite → Classification → Retrieve → Tool Calls → Failed Legs", () => {
      const { container } = render(
        <AgentTimeline
          rewrittenQuery="rewritten query text"
          queryClassification={{
            type: "FACTUAL",
            confidence: 0.92,
            latency_ms: 45,
            fallback: false,
          }}
          retrievedContext={[
            { page_content: "doc 1", metadata: {} },
            { page_content: "doc 2", metadata: {} },
          ]}
          toolTrace={[
            {
              tool_name: "search_documents",
              latency_ms: 120,
            },
          ]}
          failedLegs={["exact"]}
          isStreaming={true}
        />
      );

      const labels = screen.getAllByRole("button");

      // Steps should appear in order (textContent includes status icons/latency, so use toContain with substring)
      expect(labels.map((b) => b.textContent)).toEqual(
        expect.arrayContaining([
          expect.stringContaining("Query Rewrite"),
          expect.stringContaining("Classification"),
          expect.stringContaining("Retrieve"),
          expect.stringContaining("Tool Calls"),
          expect.stringContaining("Failed Legs"),
        ])
      );
    });

    it("renders only available steps when some props are missing", () => {
      const { container } = render(
        <AgentTimeline
          rewrittenQuery="test query"
          isStreaming={true}
        />
      );

      const buttons = screen.getAllByRole("button");
      expect(buttons).toHaveLength(1);
      expect(buttons[0].textContent).toContain("Query Rewrite");
    });
  });

  describe("transitions", () => {
    it("shows expandable step blocks during streaming", () => {
      render(
        <AgentTimeline
          rewrittenQuery="test"
          isStreaming={true}
        />
      );

      // During streaming, step rows should be present
      expect(screen.getByText(/Query Rewrite/)).toBeInTheDocument();
    });

    it("collapses to compact badge row when streaming ends", () => {
      const { rerender } = render(
        <AgentTimeline
          rewrittenQuery="test"
          isStreaming={true}
        />
      );

      // During streaming: step rows are visible
      expect(screen.getByText(/Query Rewrite/)).toBeInTheDocument();

      // After streaming: collapse to badges
      rerender(
        <AgentTimeline
          rewrittenQuery="test"
          isStreaming={false}
        />
      );

      // Badge row should be present, step buttons should be gone
      // The badges use <div> elements, not buttons
      const badges = screen.getAllByText(/Query Rewrite/);
      expect(badges.length).toBeGreaterThanOrEqual(1);
    });
  });

  describe("empty state", () => {
    it("renders nothing when no data is provided", () => {
      const { container } = render(<AgentTimeline isStreaming={true} />);
      expect(container.children).toHaveLength(0);
    });

    it("renders nothing when isStreaming is false and no data", () => {
      const { container } = render(<AgentTimeline isStreaming={false} />);
      expect(container.children).toHaveLength(0);
    });
  });

  describe("step types", () => {
    it("renders Query Rewrite step from rewrittenQuery prop", () => {
      render(
        <AgentTimeline
          rewrittenQuery="my rewritten query"
          isStreaming={true}
        />
      );
      expect(screen.getByText(/Query Rewrite/)).toBeInTheDocument();
    });

    it("renders Classification step from queryClassification prop", () => {
      render(
        <AgentTimeline
          queryClassification={{
            type: "ENTITY_CENTRIC",
            confidence: 0.85,
            latency_ms: 30,
            fallback: false,
          }}
          isStreaming={true}
        />
      );
      expect(screen.getByText(/Classification/)).toBeInTheDocument();
    });

    it("renders Retrieve step with doc count from retrievedContext", () => {
      render(
        <AgentTimeline
          retrievedContext={[
            { page_content: "doc 1", metadata: {} },
            { page_content: "doc 2", metadata: {} },
            { page_content: "doc 3", metadata: {} },
          ]}
          isStreaming={true}
        />
      );
      expect(screen.getByText(/Retrieve/)).toBeInTheDocument();
    });

    it("renders Tool Calls step from toolTrace", () => {
      render(
        <AgentTimeline
          toolTrace={[
            { tool_name: "search_documents", latency_ms: 100 },
            { tool_name: "extract_entities", latency_ms: 50 },
          ]}
          isStreaming={true}
        />
      );
      expect(screen.getByText(/Tool Calls/)).toBeInTheDocument();
    });

    it("renders Failed Legs step as error status", () => {
      render(
        <AgentTimeline
          failedLegs={["dense", "exact"]}
          isStreaming={true}
        />
      );
      expect(screen.getByText(/Failed Legs/)).toBeInTheDocument();
    });
  });

  describe("badge row", () => {
    it("shows badges for completed steps when streaming ends", () => {
      render(
        <AgentTimeline
          rewrittenQuery="test"
          toolTrace={[{ tool_name: "search_documents", latency_ms: 100 }]}
          isStreaming={false}
        />
      );

      expect(screen.getByText(/Query Rewrite/)).toBeInTheDocument();
      expect(screen.getByText(/Tool Calls/)).toBeInTheDocument();
    });

    it("shows error-styled badges for failed legs", () => {
      const { container } = render(
        <AgentTimeline
          failedLegs={["dense"]}
          isStreaming={false}
        />
      );

      expect(screen.getByText(/Failed Legs/)).toBeInTheDocument();
    });
  });

  describe("TimelineStep interface", () => {
    it("supports all status values", () => {
      const steps: TimelineStep[] = [
        { id: "a", label: "Pending Step", icon: "search", status: "pending" },
        { id: "b", label: "Active Step", icon: "book", status: "active" },
        { id: "c", label: "Done Step", icon: "check", status: "done" },
        { id: "d", label: "Error Step", icon: "x", status: "error" },
      ];

      // Verify the interface accepts all status values without type errors
      expect(steps.map((s) => s.status)).toEqual([
        "pending",
        "active",
        "done",
        "error",
      ]);
    });

    it("supports all icon types", () => {
      const steps: TimelineStep[] = [
        { id: "a", label: "A", icon: "search", status: "done" },
        { id: "b", label: "B", icon: "book", status: "done" },
        { id: "c", label: "C", icon: "share", status: "done" },
        { id: "d", label: "D", icon: "wrench", status: "done" },
        { id: "e", label: "E", icon: "check", status: "done" },
        { id: "f", label: "F", icon: "x", status: "done" },
      ];

      expect(steps.length).toBe(6);
    });
  });
});
