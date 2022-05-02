// populationGrid.test.js

import React from "react";
import { unmountComponentAtNode } from "react-dom";
import { render, screen } from '@testing-library/react';

import PopulationGrid from "./populationGrid";
let container = null;

beforeEach(() => {
  // setup a DOM element as a render target
  container = document.createElement("div");
  document.body.appendChild(container);
});

afterEach(() => {
  // cleanup on exiting
  unmountComponentAtNode(container);
  container.remove();
  container = null;
});


test('renders population grid', () => {
  render(<PopulationGrid />);
  const linkElement = screen.getByTestId(/spinner/i);
  expect(linkElement).toBeInTheDocument();
});
