/**
 * Tests for cancelStream utility.
 *
 * Verifies:
 *  1. Correct endpoint URL with Bearer token
 *  2. Returns true on 200 response
 *  3. Returns false on network error
 *  4. Returns false on non-200 response
 */
import { cancelStream } from "@/lib/cancel-stream";

// jsdom provides localStorage
const mockToken = "test-jwt-token";

beforeEach(() => {
  localStorage.setItem("token", mockToken);
  // Reset fetch mock between tests
  global.fetch = jest.fn();
});

afterEach(() => {
  jest.restoreAllMocks();
});

describe("cancelStream", () => {
  it("calls the correct endpoint URL with Bearer token", async () => {
    (global.fetch as jest.Mock).mockResolvedValue({
      ok: true,
      status: 200,
    });

    await cancelStream("42");

    expect(global.fetch).toHaveBeenCalledWith("/api/chat/42/cancel", {
      method: "POST",
      headers: {
        Authorization: `Bearer ${mockToken}`,
      },
    });
  });

  it("returns true on 200 response", async () => {
    (global.fetch as jest.Mock).mockResolvedValue({
      ok: true,
      status: 200,
    });

    const result = await cancelStream("42");

    expect(result).toBe(true);
  });

  it("returns false on network error", async () => {
    (global.fetch as jest.Mock).mockRejectedValue(new Error("Network failure"));

    const result = await cancelStream("42");

    expect(result).toBe(false);
  });

  it("returns false on non-200 response", async () => {
    (global.fetch as jest.Mock).mockResolvedValue({
      ok: false,
      status: 500,
    });

    const result = await cancelStream("42");

    expect(result).toBe(false);
  });
});
