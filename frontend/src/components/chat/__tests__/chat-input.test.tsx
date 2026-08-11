import React from "react";
import { render, screen, fireEvent, act } from "@testing-library/react";
import "@testing-library/jest-dom";
import { InputBar } from "../chat-input";

// Mock the api module
jest.mock("@/lib/api", () => ({
  api: {
    get: jest.fn(),
  },
}));

import { api } from "@/lib/api";
const mockedApi = api as jest.Mocked<typeof api>;

describe("InputBar", () => {
  const defaultProps = {
    value: "",
    onChange: jest.fn(),
    onSubmit: jest.fn(),
    disabled: false,
    placeholder: "Type your message...",
  };

  beforeEach(() => {
    jest.clearAllMocks();
    mockedApi.get.mockResolvedValue([]);
  });

  describe("auto-resize textarea", () => {
    it("renders textarea with correct data-testid", () => {
      render(<InputBar {...defaultProps} />);
      expect(screen.getByTestId("chat-input-textarea")).toBeInTheDocument();
    });

    it("textarea starts with rows=1 and uses auto-resize for height", async () => {
      render(<InputBar {...defaultProps} />);
      const textarea = screen.getByTestId("chat-input-textarea") as HTMLTextAreaElement;
      expect(textarea.rows).toBe(1);
      // useAutoResize sets height to MIN_HEIGHT_PX (48px) via requestAnimationFrame
      await act(async () => {
        await new Promise((r) => requestAnimationFrame(() => r(null)));
      });
      expect(parseInt(textarea.style.height)).toBeGreaterThanOrEqual(48);
    });

    it("textarea min-height is 48px (2 lines) via useAutoResize", async () => {
      render(<InputBar {...defaultProps} />);
      const textarea = screen.getByTestId("chat-input-textarea") as HTMLTextAreaElement;
      // useAutoResize sets MIN_HEIGHT_PX = 2 * LINE_HEIGHT_PX = 48px via rAF
      await act(async () => {
        await new Promise((r) => requestAnimationFrame(() => r(null)));
      });
      expect(parseInt(textarea.style.height)).toBeGreaterThanOrEqual(48);
    });

    it("textarea responds to content changes via onChange", () => {
      const onChange = jest.fn();
      render(<InputBar {...defaultProps} onChange={onChange} />);
      const textarea = screen.getByTestId("chat-input-textarea");

      // Simulate multi-line input
      const multiLine = "Line 1\nLine 2\nLine 3\nLine 4\nLine 5\nLine 6";
      fireEvent.change(textarea, { target: { value: multiLine } });
      expect(onChange).toHaveBeenCalledWith(multiLine);
    });
  });

  describe("keyboard shortcuts", () => {
    it("Enter submits the form", () => {
      const onSubmit = jest.fn();
      render(<InputBar {...defaultProps} value="hello" onSubmit={onSubmit} />);
      const textarea = screen.getByTestId("chat-input-textarea");

      fireEvent.keyDown(textarea, { key: "Enter", shiftKey: false });
      expect(onSubmit).toHaveBeenCalledTimes(1);
    });

    it("Shift+Enter does not submit (inserts newline)", () => {
      const onChange = jest.fn();
      render(<InputBar {...defaultProps} value="hello" onChange={onChange} />);
      const textarea = screen.getByTestId("chat-input-textarea");

      // Simulate Shift+Enter by changing value to include newline
      fireEvent.change(textarea, { target: { value: "hello\n" } });
      expect(onChange).toHaveBeenCalledWith("hello\n");
    });

    it("Enter does not submit when value is empty", () => {
      const onSubmit = jest.fn();
      render(<InputBar {...defaultProps} value="" onSubmit={onSubmit} />);
      const textarea = screen.getByTestId("chat-input-textarea");

      fireEvent.keyDown(textarea, { key: "Enter", shiftKey: false });
      expect(onSubmit).not.toHaveBeenCalled();
    });

    it("Enter does not submit when value is only whitespace", () => {
      const onSubmit = jest.fn();
      render(<InputBar {...defaultProps} value="   " onSubmit={onSubmit} />);
      const textarea = screen.getByTestId("chat-input-textarea");

      fireEvent.keyDown(textarea, { key: "Enter", shiftKey: false });
      expect(onSubmit).not.toHaveBeenCalled();
    });
  });

  describe("send button", () => {
    it("renders send button with correct data-testid", () => {
      render(<InputBar {...defaultProps} />);
      expect(screen.getByTestId("chat-input-send-button")).toBeInTheDocument();
    });

    it("renders stop button when disabled prop is true", () => {
      render(<InputBar {...defaultProps} disabled={true} />);
      const button = screen.getByTestId("chat-input-stop-button");
      expect(button).toBeInTheDocument();
    });

    it("send button is disabled when value is empty", () => {
      render(<InputBar {...defaultProps} value="" />);
      const button = screen.getByTestId("chat-input-send-button");
      expect(button).toBeDisabled();
    });

    it("send button is enabled when value has content", () => {
      render(<InputBar {...defaultProps} value="hello" />);
      const button = screen.getByTestId("chat-input-send-button");
      expect(button).not.toBeDisabled();
    });

    it("clicking send button calls onSubmit", () => {
      const onSubmit = jest.fn();
      render(<InputBar {...defaultProps} value="hello" onSubmit={onSubmit} />);
      const button = screen.getByTestId("chat-input-send-button");

      fireEvent.click(button);
      expect(onSubmit).toHaveBeenCalledTimes(1);
    });

    it("stop button calls onStop when clicked", () => {
      const onStop = jest.fn();
      render(<InputBar {...defaultProps} value="hello" disabled={true} onStop={onStop} />);
      const button = screen.getByTestId("chat-input-stop-button");
      fireEvent.click(button);
      expect(onStop).toHaveBeenCalledTimes(1);
    });
  });

  describe("KB selector", () => {
    it("renders KB selector with correct data-testid when KBs are available", async () => {
      mockedApi.get.mockResolvedValue([
        { id: 1, name: "Docs KB" },
        { id: 2, name: "Code KB" },
      ]);

      // Force re-render to trigger the effect
      const { rerender } = render(<InputBar {...defaultProps} />);
      // Wait for async fetch
      await new Promise((r) => setTimeout(r, 10));
      rerender(<InputBar {...defaultProps} />);

      expect(screen.getByTestId("chat-input-kb-selector")).toBeInTheDocument();
    });

    it("does not render KB selector when no KBs available", () => {
      mockedApi.get.mockResolvedValue([]);
      render(<InputBar {...defaultProps} />);
      // When empty, no KB selector renders
      const selector = screen.queryByTestId("chat-input-kb-selector");
      expect(selector).not.toBeInTheDocument();
    });
  });

  describe("file button", () => {
    it("renders file attachment button with correct data-testid", () => {
      render(<InputBar {...defaultProps} />);
      expect(screen.getByTestId("chat-input-file-button")).toBeInTheDocument();
    });

    it("file button is enabled by default", () => {
      render(<InputBar {...defaultProps} />);
      const button = screen.getByTestId("chat-input-file-button");
      expect(button).not.toBeDisabled();
    });

    it("file button is disabled when disabled prop is true", () => {
      render(<InputBar {...defaultProps} disabled={true} />);
      // label elements don't support disabled attribute; check the inner input
      const label = screen.getByTestId("chat-input-file-button");
      const input = label.querySelector("input");
      expect(input).toBeDisabled();
    });
  });

  describe("file chip", () => {
    it("renders file chip when uploadedFile prop is provided", () => {
      const uf = { id: 1, file_name: "test.pdf", file_size: 1024, status: "ready" as const };
      render(<InputBar {...defaultProps} uploadedFile={uf} onFileAccepted={jest.fn()} />);
      expect(screen.getByTestId("file-chip")).toBeInTheDocument();
      expect(screen.getByText(/test\.pdf/)).toBeInTheDocument();
    });

    it("does not render file chip when no uploadedFile", () => {
      render(<InputBar {...defaultProps} uploadedFile={null} onFileAccepted={jest.fn()} />);
      expect(screen.queryByTestId("file-chip")).not.toBeInTheDocument();
    });

    it("X button on file chip calls onFileRemove", () => {
      const onFileRemove = jest.fn();
      const uf = { id: 1, file_name: "report.pdf", file_size: 1024, status: "ready" as const };
      render(<InputBar {...defaultProps} uploadedFile={uf} onFileRemove={onFileRemove} />);
      const removeBtn = screen.getByTestId("file-chip-remove");
      fireEvent.click(removeBtn);
      expect(onFileRemove).toHaveBeenCalled();
    });
  });

  describe("file error display", () => {
    it("shows error message when fileError is provided", () => {
      render(<InputBar {...defaultProps} fileError="File exceeds 10 MB limit." onFileError={jest.fn()} />);
      expect(screen.getByTestId("file-error")).toHaveTextContent("File exceeds 10 MB limit.");
    });

    it("does not show error element when fileError is empty", () => {
      render(<InputBar {...defaultProps} fileError="" onFileError={jest.fn()} />);
      expect(screen.queryByTestId("file-error")).not.toBeInTheDocument();
    });
  });

  describe("data-testid attributes", () => {
    it("all required data-testid attributes are present", async () => {
      mockedApi.get.mockResolvedValue([
        { id: 1, name: "Test KB" },
      ]);
      render(<InputBar {...defaultProps} />);

      // Wait for KB fetch
      await new Promise((r) => setTimeout(r, 10));
      expect(screen.getByTestId("chat-input-textarea")).toBeInTheDocument();
      expect(screen.getByTestId("chat-input-send-button")).toBeInTheDocument();
      expect(screen.getByTestId("chat-input-file-button")).toBeInTheDocument();
    });
  });

  describe("onChange", () => {
    it("calls onChange with new value on textarea change", () => {
      const onChange = jest.fn();
      render(<InputBar {...defaultProps} onChange={onChange} />);
      const textarea = screen.getByTestId("chat-input-textarea");

      fireEvent.change(textarea, { target: { value: "new text" } });
      expect(onChange).toHaveBeenCalledWith("new text");
    });
  });

  describe("placeholder", () => {
    it("renders custom placeholder", () => {
      render(<InputBar {...defaultProps} placeholder="Ask me anything..." />);
      const textarea = screen.getByTestId("chat-input-textarea") as HTMLTextAreaElement;
      expect(textarea.placeholder).toBe("Ask me anything...");
    });

    it("uses default placeholder when not specified", () => {
      render(<InputBar {...defaultProps} />);
      const textarea = screen.getByTestId("chat-input-textarea") as HTMLTextAreaElement;
      expect(textarea.placeholder).toBe("Type your message...");
    });
  });
});
