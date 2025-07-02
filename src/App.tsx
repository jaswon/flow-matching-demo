import React, { useState, useCallback, useRef } from 'react';

import { train_step, make_data } from './logic/train';
import { generate_samples, REVERSE_STEPS } from './logic/reverse';

import PlotCanvas from './components/PlotCanvas';
import LossPlotCanvas from './components/LossPlotCanvas';


export default () => {
  const [batchIdx, setBatchIdx] = useState(0);
  const [sliderValue, setSliderValue] = useState(0);
  const [lossHistory, setLossHistory] = useState<number[]>([]);
  const [sampleTrajs, setSampleTrajs] = useState<[number,number][][]>([]);
  const [X] = useState(() => make_data(1024));
  const trainingRef = useRef(false);

  const noSamples = sampleTrajs.length == 0;
  const loadingSamples = !trainingRef.current && noSamples;

  // Generate samples for reverse process
  const generateSamples = useCallback(async () => {
    setSampleTrajs(await generate_samples());
  }, []);

  // Generate samples on initial mount
  React.useEffect(() => {
    generateSamples();
  }, []);

  // Simplified training loop
  const startTraining = useCallback(async () => {
    setSampleTrajs([]);
    setSliderValue(0);
    trainingRef.current = true;
    while (trainingRef.current) {
      const loss = await train_step();
      setLossHistory(hist => [...hist, loss.dataSync()[0]]);
      setBatchIdx(idx => idx + 1);
      await new Promise(res => setTimeout(res, 0)); // Allow UI updates
    }
    generateSamples();
  }, []);

  const stopTraining = useCallback(() => {
    trainingRef.current = false;
  }, []);

  // Button click handler
  const handleToggle = () => {
    if (trainingRef.current) {
      stopTraining();
      return;
    }
    if (noSamples) return;
    startTraining();
  };

  return (
    <div id="container">
      <h2>Flow Matching Demo</h2>
      <div id="controls">
        <button
          id="toggleTrainBtn"
          className={loadingSamples ? 'loading' : ''}
          disabled={loadingSamples}
          onClick={handleToggle}
        >
          <span className="btn-label">{(() => {
            if (trainingRef.current || noSamples) return 'Pause';
            if (batchIdx > 0) return 'Resume Training';
            return 'Start Training';
          })()}</span>
          <span className="spinner"></span>
        </button>
        <div className="slider-container">
          <span id="sliderLabel">
            Reverse Step: {sliderValue.toString().padStart(2)}/{REVERSE_STEPS}
          </span>
          <input
            id="stepSlider"
            type="range"
            min={0}
            max={REVERSE_STEPS}
            value={sliderValue}
            onChange={(e: React.ChangeEvent<HTMLInputElement>) => {
              setSliderValue(Number(e.target.value));
            }}
            disabled={trainingRef.current || noSamples}
          />
        </div>
      </div>
      <PlotCanvas X={X} sampleTrajs={sampleTrajs} sliderValue={sliderValue} />
      <LossPlotCanvas lossHistory={lossHistory} batchIdx={batchIdx} />
    </div>
  );
}