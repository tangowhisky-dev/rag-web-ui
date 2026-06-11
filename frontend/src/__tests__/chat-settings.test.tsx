import React from "react";
import { render, screen, fireEvent } from "@testing-library/react";
import ChatSettings from "@/components/chat/chat-settings";

// Mock the api module
jest.mock("@/lib/api", () => ({
  api: {
    patch: jest.fn().mockResolvedValue({}),
  },
}));

const mockChat = {
  id: 1,
  title: "Test Chat",
  temperature: 0.7,
  model_name: "gpt-4o",
  use_dense: true,
  use_sparse: true,
  use_exact: false,
  use_graph_rag: false,
};

describe("ChatSettings", () => {
  it("renders the temperature slider", () => {
    render(
      <ChatSettings chat={mockChat} onClose={jest.fn()} onUpdate={jest.fn()} />
    );
    expect(screen.getByTestId("temperature-slider")).toBeInTheDocument();
    expect(screen.getByTestId("temperature-value")).toHaveTextContent("0.7");
  });

  it("renders all retrieval leg toggles", () => {
    render(
      <ChatSettings chat={mockChat} onClose={jest.fn()} onUpdate={jest.fn()} />
    );
    expect(screen.getByTestId("toggle-vector")).toBeInTheDocument();
    expect(screen.getByTestId("toggle-exact")).toBeInTheDocument();
    expect(screen.getByTestId("toggle-graph")).toBeInTheDocument();
  });

  it("toggles a retrieval leg when clicked", () => {
    render(
      <ChatSettings chat={mockChat} onClose={jest.fn()} onUpdate={jest.fn()} />
    );
    const toggle = screen.getByTestId("toggle-vector");
    expect(toggle).toHaveAttribute("aria-checked", "true");
    fireEvent.click(toggle);
    expect(toggle).toHaveAttribute("aria-checked", "false");
  });

  it("updates temperature value when slider changes", () => {
    render(
      <ChatSettings chat={mockChat} onClose={jest.fn()} onUpdate={jest.fn()} />
    );
    const slider = screen.getByTestId("temperature-slider");
    fireEvent.change(slider, { target: { value: "0.3" } });
    expect(screen.getByTestId("temperature-value")).toHaveTextContent("0.3");
  });

  it("renders the model selector with correct default", () => {
    render(
      <ChatSettings chat={mockChat} onClose={jest.fn()} onUpdate={jest.fn()} />
    );
    const select = screen.getByTestId("model-select") as HTMLSelectElement;
    expect(select.value).toBe("gpt-4o");
  });
});
