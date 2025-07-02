import { setupUI } from './ui.js';
import { plot_loss } from './plot_loss.js';
import { clear_plot } from './plot.js';

window.addEventListener('DOMContentLoaded', () => {
  clear_plot();
  setupUI(plot_loss);
  plot_loss();
});
