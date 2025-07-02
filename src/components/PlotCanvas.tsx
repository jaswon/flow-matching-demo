import React, { useRef, useEffect } from 'react';

interface PlotCanvasProps {
  X: [number,number][];
  sampleTrajs: [number, number][][];
  sliderValue: number;
}

function to_canvas(
  canvas: HTMLCanvasElement, 
  x: number, 
  y: number,
): [number, number] {
  return [
    canvas.width / 2 * (1 + x / 5),
    canvas.height / 2 * (1 + y / 5),
  ];
} 

function plot_points(
  ctx: CanvasRenderingContext2D, 
  canvas: HTMLCanvasElement, 
  points: [number,number][], 
  color: string, 
  radius: number,
) {
  for (const [x, y] of points) {
    const [px, py] = to_canvas(canvas, x, y);
    ctx.beginPath();
    ctx.arc(px, py, radius, 0, 2 * Math.PI);
    ctx.fillStyle = color;
    ctx.fill();
  }
}

const PlotCanvas: React.FC<PlotCanvasProps> = ({ X, sampleTrajs, sliderValue }) => {
  const canvasRef = useRef<HTMLCanvasElement>(null);

  useEffect(() => {
    const canvas = canvasRef.current;
    if (!canvas) return;
    const ctx = canvas.getContext('2d');
    if (!ctx) return;

    plot_points(ctx, canvas, X, 'rgba(0,0,255,.1)', 2);
    for (let traj of sampleTrajs) {
      plot_points(ctx, canvas, [traj[sliderValue]], 'red', 4);
    }
    return () => ctx.clearRect(0, 0, canvas.width, canvas.height);
  }, [X, sampleTrajs, sliderValue]);

  return <canvas ref={canvasRef} width={450} height={450} style={{ marginBottom: 18 }} />;
};

export default PlotCanvas; 