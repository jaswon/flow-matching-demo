// Plotting utilities and color constants

import { X } from "./train";

const canvas = document.getElementById('plot');
const ctx = canvas.getContext("2d");

function plot_points(points, color, radius) {
  for (const [x, y] of points) {
    const [px, py] = to_canvas(x, y);
    ctx.beginPath();
    ctx.arc(px, py, radius, 0, 2 * Math.PI);
    ctx.fillStyle = color;
    ctx.fill();
  }
}

export function plot_pred_points(points) {
  plot_points(points, 'red', 4)
}

export function clear_plot() {
  ctx.clearRect(0, 0, canvas.width, canvas.height);
  plot_points(X, 'rgba(0,0,255,.1)', 2);
}

export function to_canvas(x, y) {
  return [
    canvas.width / 2 * (1 + x / 5),
    canvas.height / 2 * (1 + y / 5),
  ];
} 