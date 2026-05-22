import React from 'react';
import { render, screen, waitFor } from '@testing-library/react';
import '@testing-library/jest-dom';

// Mock mermaid module for dynamic import inside useEffect
jest.mock('mermaid', () => {
  return {
    __esModule: true,
    default: {
      initialize: jest.fn(),
      render: jest.fn(),
    },
  };
});

import mermaid from 'mermaid';
import MermaidDiagram from '../mermaid-diagram';

const mockMermaidRender = mermaid.render as jest.Mock;

describe('MermaidDiagram', () => {
  beforeEach(() => {
    jest.clearAllMocks();
  });

  it('renders SVG output when mermaid.render resolves', async () => {
    mockMermaidRender.mockResolvedValue({ svg: '<svg><text>graph</text></svg>' });

    render(<MermaidDiagram code="graph TD; A-->B;" />);

    await waitFor(() => {
      const container = document.querySelector('.mermaid-diagram');
      expect(container).toBeInTheDocument();
    });

    const container = document.querySelector('.mermaid-diagram');
    expect(container!.innerHTML).toContain('<svg>');
  });

  it('renders error block when mermaid.render throws', async () => {
    mockMermaidRender.mockRejectedValue(new Error('parse error'));

    render(<MermaidDiagram code="invalid diagram %%%" />);

    await waitFor(() => {
      expect(screen.getByText(/Mermaid error:/i)).toBeInTheDocument();
    });
  });

  it('shows loading state before render resolves', () => {
    // Never resolve so we stay in loading state
    mockMermaidRender.mockReturnValue(new Promise(() => {}));

    render(<MermaidDiagram code="graph TD; A-->B;" />);

    // Before useEffect resolves, shows loading pulse div
    const loadingEl = document.querySelector('.animate-pulse');
    expect(loadingEl).toBeInTheDocument();
  });
});
