import React, { useRef, useEffect } from 'react';

export function plot_loss(
  ctx: CanvasRenderingContext2D,
  canvas: HTMLCanvasElement,
  loss_history: number[],
  batch_idx: number
) {
  ctx.clearRect(0, 0, canvas.width, canvas.height);
  // Draw axes
  ctx.strokeStyle = '#aaa';
  ctx.beginPath();
  ctx.moveTo(40, 10);
  ctx.lineTo(40, 180);
  ctx.lineTo(490, 180);
  ctx.stroke();
  if (loss_history.length < 2) return;
  let minEpoch = 1;
  let maxEpoch = loss_history.length;

  // Aggregate into log bins
  const numBins = 100;
  const logMinEpoch = Math.log10(minEpoch);
  const logMaxEpoch = Math.log10(maxEpoch);
  let bins: {epoch: number, loss: number}[][] = Array.from({length: numBins}, () => []);
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
  let binPoints: {epoch: number, loss: number}[] = [];
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
  ctx.strokeStyle = 'red';
  ctx.beginPath();
  for (let i = 0; i < binPoints.length; ++i) {
    let {epoch, loss} = binPoints[i];
    let logEpoch = Math.log10(epoch);
    let x = 40 + 450 * (logEpoch - logMinEpoch) / (logMaxEpoch - logMinEpoch);
    let logLoss = Math.log10(loss);
    let logMinLoss = Math.log10(minLoss);
    let logMaxLoss = Math.log10(maxLoss);
    let y = 180 - 160 * (logLoss - logMinLoss) / (logMaxLoss - logMinLoss);
    if (i === 0) ctx.moveTo(x, y);
    else ctx.lineTo(x, y);
  }
  ctx.stroke();
  // Draw min/max labels (log scale)
  ctx.fillStyle = '#333';
  ctx.font = '12px sans-serif';
  ctx.fillText(maxLoss.toExponential(2), 5, 20);
  ctx.fillText(minLoss.toExponential(2), 5, 180);
  ctx.fillText('Loss (log-log)', 350, 20);
  ctx.fillText(`Batch (current: ${batch_idx})`, 350, 195);
} 

interface LossPlotCanvasProps {
  lossHistory: number[];
  batchIdx: number;
}

const LossPlotCanvas: React.FC<LossPlotCanvasProps> = ({ lossHistory, batchIdx }) => {
  const canvasRef = useRef<HTMLCanvasElement>(null);

  useEffect(() => {
    const canvas = canvasRef.current;
    if (!canvas) return;
    const ctx = canvas.getContext('2d');
    if (!ctx) return;
    
    plot_loss(ctx, canvas, lossHistory, batchIdx);
  }, [lossHistory, batchIdx]);

  return <canvas ref={canvasRef} width={500} height={200} />;
};

export default LossPlotCanvas; 