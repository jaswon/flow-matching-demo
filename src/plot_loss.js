import { loss_history, batch_idx } from './train.js';

const lossCanvas = document.getElementById('lossPlot');
const lossCtx = lossCanvas.getContext('2d');

export function plot_loss() {
  lossCtx.clearRect(0, 0, lossCanvas.width, lossCanvas.height);
  // Draw axes
  lossCtx.strokeStyle = '#aaa';
  lossCtx.beginPath();
  lossCtx.moveTo(40, 10);
  lossCtx.lineTo(40, 180);
  lossCtx.lineTo(490, 180);
  lossCtx.stroke();
  if (loss_history.length < 2) return;
  let minEpoch = 1;
  let maxEpoch = loss_history.length;

  // Aggregate into log bins
  const numBins = 100;
  const logMinEpoch = Math.log10(minEpoch);
  const logMaxEpoch = Math.log10(maxEpoch);
  let bins = Array.from({length: numBins}, () => []);
  for (let i = 0; i < loss_history.length; ++i) {
    let lossVal = loss_history[i];
    if (lossVal <= 0) continue;
    let logEpoch = Math.log10(i + 1);
    let binIdx = Math.floor((logEpoch - logMinEpoch) / (logMaxEpoch - logMinEpoch) * numBins);
    if (binIdx < 0) binIdx = 0;
    if (binIdx >= numBins) binIdx = numBins - 1;
    bins[binIdx].push({epoch: i + 1, loss: lossVal});
  }
  // Compute median per bin
  let binPoints = [];
  for (let b = 0; b < numBins; ++b) {
    if (bins[b].length === 0) continue;
    let epochs = bins[b].map(v => v.epoch);
    let losses = bins[b].map(v => v.loss);
    epochs.sort((a, b) => a - b);
    losses.sort((a, b) => a - b);
    // Median epoch
    let mid = Math.floor(epochs.length / 2);
    let medianEpoch = epochs.length % 2 === 0 ? (epochs[mid - 1] + epochs[mid]) / 2 : epochs[mid];
    // Median loss
    let medianLoss = losses.length % 2 === 0 ? (losses[mid - 1] + losses[mid]) / 2 : losses[mid];
    binPoints.push({epoch: medianEpoch, loss: medianLoss});
  }

  // Outlier trimming on binned median losses
  let medianLosses = binPoints.map(p => p.loss).filter(l => l > 0);
  medianLosses.sort((a, b) => a - b);
  const trim = Math.floor(medianLosses.length * 0.05);
  if (medianLosses.length > 2 * trim) {
    medianLosses = medianLosses.slice(trim, medianLosses.length - trim);
  }
  let minLoss = Math.min(...medianLosses);
  let maxLoss = Math.max(...medianLosses);
  if (minLoss === maxLoss) maxLoss += 1e-6;

  // Plot loss curve (log-log)
  lossCtx.strokeStyle = 'red';
  lossCtx.beginPath();
  for (let i = 0; i < binPoints.length; ++i) {
    let {epoch, loss} = binPoints[i];
    let logEpoch = Math.log10(epoch);
    let x = 40 + 450 * (logEpoch - logMinEpoch) / (logMaxEpoch - logMinEpoch);
    let logLoss = Math.log10(loss);
    let logMinLoss = Math.log10(minLoss);
    let logMaxLoss = Math.log10(maxLoss);
    let y = 180 - 160 * (logLoss - logMinLoss) / (logMaxLoss - logMinLoss);
    if (i === 0) lossCtx.moveTo(x, y);
    else lossCtx.lineTo(x, y);
  }
  lossCtx.stroke();
  // Draw min/max labels (log scale)
  lossCtx.fillStyle = '#333';
  lossCtx.font = '12px sans-serif';
  lossCtx.fillText(maxLoss.toExponential(2), 5, 20);
  lossCtx.fillText(minLoss.toExponential(2), 5, 180);
  lossCtx.fillText('Loss (log-log)', 350, 20);
  lossCtx.fillText(`Batch (current: ${batch_idx})`, 350, 195);
} 