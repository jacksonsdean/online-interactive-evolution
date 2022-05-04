// IECApp.test.js

import React from "react";
import { unmountComponentAtNode } from "react-dom";
import { render, screen } from '@testing-library/react';
import IECApp  from "./IECApp";

let container = null;

// fix for react-sliders
import { install } from "resize-observer";
install();


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


test('renders welcome text', () => {
  render(<IECApp />);
  const linkElement = screen.getByText(/Interactive Evolutionary Art/i);
  expect(linkElement).toBeInTheDocument();
});
