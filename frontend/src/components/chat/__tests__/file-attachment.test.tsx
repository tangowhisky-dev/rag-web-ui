import React from "react";
import { render, screen, fireEvent } from "@testing-library/react";
import "@testing-library/jest-dom";
import { FileChip, MAX_FILE_SIZE, SUPPORTED_TYPES } from "../file-attachment";

// ── helpers ────────────────────────────────────────────────────────────────────

function makeFile(name: string, type: string, sizeBytes = 1024): File {
  const content = new Uint8Array(sizeBytes);
  return new File([content], name, { type });
}

// ── FileChip ───────────────────────────────────────────────────────────────────

describe("FileChip", () => {
  it("renders the filename", () => {
    const file = makeFile("report.pdf", "application/pdf");
    render(<FileChip uploadedFile={{ id: 1, file_name: file.name, file_size: file.size, status: "ready" }} onRemove={jest.fn()} />);
    expect(screen.getByTestId("file-chip")).toBeInTheDocument();
    expect(screen.getByText(/report\.pdf/)).toBeInTheDocument();
  });

  it("renders formatted file size in bytes", () => {
    const file = makeFile("tiny.txt", "text/plain", 512);
    render(<FileChip uploadedFile={{ id: 1, file_name: file.name, file_size: file.size, status: "ready" }} onRemove={jest.fn()} />);
    expect(screen.getByText(/512 B/)).toBeInTheDocument();
  });

  it("renders formatted file size in KB", () => {
    const file = makeFile("small.txt", "text/plain", 2048);
    render(<FileChip uploadedFile={{ id: 1, file_name: file.name, file_size: file.size, status: "ready" }} onRemove={jest.fn()} />);
    expect(screen.getByText(/2\.0 KB/)).toBeInTheDocument();
  });

  it("renders formatted file size in MB", () => {
    const file = makeFile("big.pdf", "application/pdf", 2 * 1024 * 1024);
    render(<FileChip uploadedFile={{ id: 1, file_name: file.name, file_size: file.size, status: "ready" }} onRemove={jest.fn()} />);
    expect(screen.getByText(/2\.0 MB/)).toBeInTheDocument();
  });

  it("calls onRemove when X button is clicked", () => {
    const onRemove = jest.fn();
    const file = makeFile("doc.docx", "application/vnd.openxmlformats-officedocument.wordprocessingml.document");
    render(<FileChip uploadedFile={{ id: 1, file_name: file.name, file_size: file.size, status: "ready" }} onRemove={onRemove} />);
    fireEvent.click(screen.getByTestId("file-chip-remove"));
    expect(onRemove).toHaveBeenCalledTimes(1);
  });

  it("truncates long filenames", () => {
    const longName = "a".repeat(40) + ".pdf";
    const file = makeFile(longName, "application/pdf");
    render(<FileChip uploadedFile={{ id: 1, file_name: file.name, file_size: file.size, status: "ready" }} onRemove={jest.fn()} />);
    // Full name is on the chip's title attribute; display text is truncated
    const chip = screen.getByTestId("file-chip");
    expect(chip.title).toBe(longName);
    const nameSpan = chip.querySelector("span.truncate") as HTMLElement;
    expect(nameSpan.textContent!.length).toBeLessThan(longName.length);
  });
});

// ── Constants ──────────────────────────────────────────────────────────────────

describe("constants", () => {
  it("MAX_FILE_SIZE is 10 MB", () => {
    expect(MAX_FILE_SIZE).toBe(10 * 1024 * 1024);
  });

  it("SUPPORTED_TYPES includes pdf, docx, txt, csv", () => {
    expect(SUPPORTED_TYPES["application/pdf"]).toEqual([".pdf"]);
    expect(SUPPORTED_TYPES["application/vnd.openxmlformats-officedocument.wordprocessingml.document"]).toEqual([".docx"]);
    expect(SUPPORTED_TYPES["text/plain"]).toEqual([".txt"]);
    expect(SUPPORTED_TYPES["text/csv"]).toEqual([".csv"]);
  });

  it("SUPPORTED_TYPES includes images and common office formats", () => {
    expect(SUPPORTED_TYPES["image/jpeg"]).toBeDefined();
    expect(SUPPORTED_TYPES["image/png"]).toBeDefined();
    expect(SUPPORTED_TYPES["application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"]).toEqual([".xlsx"]);
  });
});
