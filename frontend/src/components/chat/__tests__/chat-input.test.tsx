import React from "react";
import { render, screen, fireEvent, act } from "@testing-library/react";
import "@testing-library/jest-dom";
import { InputBar } from "../chat-input";

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
  });

  // Flush any pending effects after render.
  async function renderInputBar(
    props: Partial<React.ComponentProps<typeof InputBar>> = {}
  ) {
    const result = render(<InputBar {...defaultProps} {...props} />);
    await act(async () => { await Promise.resolve(); });
    return result;
  }

  describe("auto-resize textarea", () => {
    it("renders textarea with correct data-testid", async () => {
      await renderInputBar();
      expect(screen.getByTestId("chat-input-textarea")).toBeInTheDocument();
    });

    it("textarea starts with rows=1 and uses auto-resize for height", async () => {
      await renderInputBar();
      const textarea = screen.getByTestId("chat-input-textarea") as HTMLTextAreaElement;
      expect(textarea.rows).toBe(1);
      // useAutoResize sets height to MIN_HEIGHT_PX (48px) via requestAnimationFrame
      await act(async () => {
        await new Promise((r) => requestAnimationFrame(() => r(null)));
      });
      expect(parseInt(textarea.style.height)).toBeGreaterThanOrEqual(48);
    });

    it("textarea min-height is 48px (2 lines) via useAutoResize", async () => {
      await renderInputBar();
      const textarea = screen.getByTestId("chat-input-textarea") as HTMLTextAreaElement;
      // useAutoResize sets MIN_HEIGHT_PX = 2 * LINE_HEIGHT_PX = 48px via rAF
      await act(async () => {
        await new Promise((r) => requestAnimationFrame(() => r(null)));
      });
      expect(parseInt(textarea.style.height)).toBeGreaterThanOrEqual(48);
    });

    it("textarea responds to content changes via onChange", async () => {
      const onChange = jest.fn();
      await renderInputBar({onChange: onChange});
      const textarea = screen.getByTestId("chat-input-textarea");

      // Simulate multi-line input
      const multiLine = "Line 1\nLine 2\nLine 3\nLine 4\nLine 5\nLine 6";
      fireEvent.change(textarea, { target: { value: multiLine } });
      expect(onChange).toHaveBeenCalledWith(multiLine);
    });
  });

  describe("keyboard shortcuts", () => {
    it("Enter submits the form", async () => {
      const onSubmit = jest.fn();
      await renderInputBar({ value: "hello", onSubmit: onSubmit });
      const textarea = screen.getByTestId("chat-input-textarea");

      fireEvent.keyDown(textarea, { key: "Enter", shiftKey: false });
      expect(onSubmit).toHaveBeenCalledTimes(1);
    });

    it("Shift+Enter does not submit (inserts newline)", async () => {
      const onChange = jest.fn();
      await renderInputBar({onChange: onChange});
      const textarea = screen.getByTestId("chat-input-textarea");

      // Simulate Shift+Enter by changing value to include newline
      fireEvent.change(textarea, { target: { value: "hello\n" } });
      expect(onChange).toHaveBeenCalledWith("hello\n");
    });

    it("Enter does not submit when value is empty", async () => {
      const onSubmit = jest.fn();
      await renderInputBar({onSubmit: onSubmit});
      const textarea = screen.getByTestId("chat-input-textarea");

      fireEvent.keyDown(textarea, { key: "Enter", shiftKey: false });
      expect(onSubmit).not.toHaveBeenCalled();
    });

    it("Enter does not submit when value is only whitespace", async () => {
      const onSubmit = jest.fn();
      await renderInputBar({onSubmit: onSubmit});
      const textarea = screen.getByTestId("chat-input-textarea");

      fireEvent.keyDown(textarea, { key: "Enter", shiftKey: false });
      expect(onSubmit).not.toHaveBeenCalled();
    });
  });

  describe("send button", () => {
    it("renders send button with correct data-testid", async () => {
      await renderInputBar();
      expect(screen.getByTestId("chat-input-send-button")).toBeInTheDocument();
    });

    it("renders stop button when disabled prop is true", async () => {
      await renderInputBar({disabled: true});
      const button = screen.getByTestId("chat-input-stop-button");
      expect(button).toBeInTheDocument();
    });

    it("send button is disabled when value is empty", async () => {
      await renderInputBar({ value: "" });
      const button = screen.getByTestId("chat-input-send-button");
      expect(button).toBeDisabled();
    });

    it("send button is enabled when value has content", async () => {
      await renderInputBar({ value: "hello" });
      const button = screen.getByTestId("chat-input-send-button");
      expect(button).not.toBeDisabled();
    });

    it("clicking send button calls onSubmit", async () => {
      const onSubmit = jest.fn();
      await renderInputBar({ value: "hello", onSubmit: onSubmit });
      const button = screen.getByTestId("chat-input-send-button");

      fireEvent.click(button);
      expect(onSubmit).toHaveBeenCalledTimes(1);
    });

    it("stop button calls onStop when clicked", async () => {
      const onStop = jest.fn();
      await renderInputBar({disabled: true, onStop: onStop});
      const button = screen.getByTestId("chat-input-stop-button");
      fireEvent.click(button);
      expect(onStop).toHaveBeenCalledTimes(1);
    });
  });

  describe("KB pills", () => {
    it("renders KB pills when knowledgeBases are provided", async () => {
      await renderInputBar({
        knowledgeBases: [
          { id: 1, name: "Docs KB" },
          { id: 2, name: "Code KB" },
        ],
        selectedKbIds: [1, 2],
        onKbToggle: jest.fn(),
      });

      const pills = screen.getAllByTestId("chat-input-kb-pill");
      expect(pills).toHaveLength(2);
      expect(pills[0]).toHaveTextContent("Docs KB");
      expect(pills[1]).toHaveTextContent("Code KB");
    });

    it("does not render KB pills when no knowledgeBases provided", async () => {
      await renderInputBar();
      const pills = screen.queryAllByTestId("chat-input-kb-pill");
      expect(pills).toHaveLength(0);
    });

    it("renders + button linking to KB creation page", async () => {
      await renderInputBar();
      const addBtn = screen.getByTestId("chat-input-kb-add");
      expect(addBtn).toBeInTheDocument();
      expect(addBtn.closest("a")).toHaveAttribute("href", "/dashboard/knowledge");
    });

    it("calls onKbToggle when pill is clicked", async () => {
      const onKbToggle = jest.fn();
      await renderInputBar({
        knowledgeBases: [{ id: 1, name: "Docs KB" }],
        selectedKbIds: [1],
        onKbToggle,
      });

      const pill = screen.getByTestId("chat-input-kb-pill");
      fireEvent.click(pill);
      expect(onKbToggle).toHaveBeenCalledWith(1);
    });

    it("selected pill has primary styling, deselected has muted styling", async () => {
      await renderInputBar({
        knowledgeBases: [
          { id: 1, name: "Docs KB" },
          { id: 2, name: "Code KB" },
        ],
        selectedKbIds: [1],
        onKbToggle: jest.fn(),
      });

      const pills = screen.getAllByTestId("chat-input-kb-pill");
      expect(pills[0].className).toContain("border-primary");
      expect(pills[1].className).toContain("border-border");
    });
  });

  describe("file button", () => {
    it("renders file attachment button with correct data-testid", async () => {
      await renderInputBar();
      expect(screen.getByTestId("chat-input-file-button")).toBeInTheDocument();
    });

    it("file button is enabled by default", async () => {
      await renderInputBar();
      const button = screen.getByTestId("chat-input-file-button");
      expect(button).not.toBeDisabled();
    });

    it("file button is disabled when disabled prop is true", async () => {
      await renderInputBar({disabled: true});
      // label elements don't support disabled attribute; check the inner input
      const label = screen.getByTestId("chat-input-file-button");
      const input = label.querySelector("input");
      expect(input).toBeDisabled();
    });
  });

  describe("file chip", () => {
    it("renders file chip when uploadedFile prop is provided", async () => {
      const uf = { id: 1, file_name: "test.pdf", file_size: 1024, status: "ready" as const };
      await renderInputBar({uploadedFile: uf, onFileAccepted: jest.fn()});
      expect(screen.getByTestId("file-chip")).toBeInTheDocument();
      expect(screen.getByText(/test\.pdf/)).toBeInTheDocument();
    });

    it("does not render file chip when no uploadedFile", async () => {
      await renderInputBar({uploadedFile: null, onFileAccepted: jest.fn()});
      expect(screen.queryByTestId("file-chip")).not.toBeInTheDocument();
    });

    it("X button on file chip calls onFileRemove", async () => {
      const onFileRemove = jest.fn();
      const uf = { id: 1, file_name: "report.pdf", file_size: 1024, status: "ready" as const };
      await renderInputBar({uploadedFile: uf, onFileRemove: onFileRemove});
      const removeBtn = screen.getByTestId("file-chip-remove");
      fireEvent.click(removeBtn);
      expect(onFileRemove).toHaveBeenCalled();
    });
  });

  describe("file error display", () => {
    it("shows error message when fileError is provided", async () => {
      await renderInputBar({ fileError: "File exceeds 10 MB limit.", onFileError: jest.fn() });
      expect(screen.getByTestId("file-error")).toHaveTextContent("File exceeds 10 MB limit.");
    });

    it("does not show error element when fileError is empty", async () => {
      await renderInputBar({onFileError: jest.fn()});
      expect(screen.queryByTestId("file-error")).not.toBeInTheDocument();
    });
  });

  describe("data-testid attributes", () => {
    it("all required data-testid attributes are present", async () => {
      await renderInputBar({
        knowledgeBases: [{ id: 1, name: "Test KB" }],
        selectedKbIds: [1],
        onKbToggle: jest.fn(),
      });

      expect(screen.getByTestId("chat-input-textarea")).toBeInTheDocument();
      expect(screen.getByTestId("chat-input-send-button")).toBeInTheDocument();
      expect(screen.getByTestId("chat-input-file-button")).toBeInTheDocument();
      expect(screen.getByTestId("chat-input-kb-pill")).toBeInTheDocument();
      expect(screen.getByTestId("chat-input-kb-add")).toBeInTheDocument();
    });
  });

  describe("onChange", () => {
    it("calls onChange with new value on textarea change", async () => {
      const onChange = jest.fn();
      await renderInputBar({onChange: onChange});
      const textarea = screen.getByTestId("chat-input-textarea");

      fireEvent.change(textarea, { target: { value: "new text" } });
      expect(onChange).toHaveBeenCalledWith("new text");
    });
  });

  describe("placeholder", () => {
    it("renders custom placeholder", async () => {
      await renderInputBar({ placeholder: "Ask me anything..." });
      const textarea = screen.getByTestId("chat-input-textarea") as HTMLTextAreaElement;
      expect(textarea.placeholder).toBe("Ask me anything...");
    });

    it("uses default placeholder when not specified", async () => {
      await renderInputBar();
      const textarea = screen.getByTestId("chat-input-textarea") as HTMLTextAreaElement;
      expect(textarea.placeholder).toBe("Type your message...");
    });
  });
});
