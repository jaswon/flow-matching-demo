// --- Data Generation ---
function make_data(n_samples) {
    const n = 4;
    const w = 4;
    const X = [];
    for (let i = 0; i < n_samples; ++i) {
        const x = Math.random()*n;
        const y = Math.random()*n/2;
        const y_int = Math.floor(y);
        const y_frac = y - y_int;
        X.push([
            w/2 * (-n/2 + x), 
            w/2 * (-n/2 + 2*y_int+y_frac+(Math.floor(x)%2)),
        ]);
    }
    return X;
}

// --- Plotting ---
const canvas = document.getElementById('plot');
const ctx = canvas.getContext('2d');
const TRUE_COLOR = 'rgba(0,0,255,.1)';
const PRED_COLOR = 'red';
function plot_points(points, color, radius = 2) {
  for (const [x, y] of points) {
    const [px, py] = to_canvas(x, y);
    ctx.beginPath();
    ctx.arc(px, py, radius, 0, 2 * Math.PI);
    ctx.fillStyle = color;
    ctx.fill();
  }
}
function clear_plot() {
  ctx.clearRect(0, 0, canvas.width, canvas.height);
}
function to_canvas(x, y) {
  return [
    canvas.width / 2 * (1 + x / 5),
    canvas.height / 2 * (1 + y / 5),
  ];
}

// --- Random Fourier Feature Embedding Layer ---
class RandomFourierFeatureEmbedding extends tf.layers.Layer {
    static className = 'RandomFourierFeatureEmbedding';
    
    constructor(dim, sigma = 1.0, config = {}) {
        super(config);
        this.dim = dim;
        this.sigma = sigma;
        this.built = false;
    }

    build(inputShape) {
        const halfDim = Math.floor(this.dim / 2);
        
        // Random frequencies from normal distribution
        this.frequencies = this.addWeight(
            'frequencies',
            [inputShape[1], halfDim],
            'float32',
            tf.initializers.randomNormal({stddev: 1.0 / this.sigma}),
            null,
            false,
        );
        
        // Random phases from uniform distribution [0, 2π]
        this.phases = this.addWeight(
            'phases',
            [halfDim],
            'float32',
            tf.initializers.randomUniform({minval: 0, maxval: 2 * Math.PI}),
            null,
            false,
        );
        
        this.built = true;
    }

    computeOutputShape(inputShape) {
        return [inputShape[0], this.dim];
    }
    
    call(inputs) {
        return tf.tidy(() => {
            const projected = tf.matMul(inputs[0], this.frequencies.read());
            const shifted = tf.add(projected, tf.expandDims(this.phases.read(), 0));
            return tf.concat([tf.cos(shifted), tf.sin(shifted)], -1); // [batch, dim]
        });
    }
    
    getConfig() {
        const config = super.getConfig();
        config.dim = this.dim;
        config.sigma = this.sigma;
        return config;
    }
}
// Register the custom layer
tf.serialization.registerClass(RandomFourierFeatureEmbedding);

// --- Model ---
const optimizer = tf.train.adam(0.0001);
const model = (() => {
    const hDim = 128;
    const xt = tf.input({shape: [2]});
    const t = tf.input({shape: [1]});
    
    let tEmb = new RandomFourierFeatureEmbedding(64, 1.0).apply(t);
    tEmb = tf.layers.dense({units:256, activation:'gelu'}).apply(tEmb);
    let x = tf.layers.dense({units:hDim}).apply(xt);
    
    for (let i = 0; i < 6; i++) {
        let r = tf.layers.layerNormalization().apply(x);
        r = tf.layers.multiply().apply([
            tf.layers.dense({units:hDim}).apply(r),
            tf.layers.dense({units:hDim, activation:'gelu'}).apply(r),
        ])
        r = tf.layers.dropout({rate:.3}).apply(r);
        r = tf.layers.dense({units:hDim}).apply(r);
        const alpha = tf.layers.dense({units:hDim}).apply(tEmb);
        r = tf.layers.multiply().apply([r, alpha]);
        x = tf.layers.add().apply([x,r]);
    }
    const output = tf.layers.dense({units: 2}).apply(x);
    return tf.model({inputs: [xt, t], outputs: output});
})();
model.summary();

// --- Prior distribution ---
function make_batch(batch_size) {
    // const W = 8;
    // return tf.randomUniform([batch_size, 2], -W/2, W/2);
    const STD = 1;
    return tf.randomNormal([batch_size, 2], 0, STD);
}

// --- Training Loop ---
let isTraining = false, batch_idx = 0, loss = 0;
const X = make_data(1024);

const lossCanvas = document.getElementById('lossPlot');
const lossCtx = lossCanvas.getContext('2d');

// Store loss history
let loss_history = [];

function plot_loss() {
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

// --- Flow Matching Loss ---
function flow_matching_loss(model, x0, x1, t) {
  // x0: data, x1: noise, t: [0,1]
  // x_t = t * x1 + (1-t) * x0
  // v_target = x1 - x0
  return tf.tidy(() => {
    const x0t = tf.tensor2d(x0);
    const x1t = tf.tensor2d(x1);
    const t_col = tf.tensor2d(t, [t.length, 1]);
    const xt = tf.add(tf.mul(t_col, x1t), tf.mul(tf.sub(1, t_col), x0t));
    const v_target = tf.sub(x1t, x0t);
    const v_pred = model.predict([xt, t_col]);
    return tf.losses.meanSquaredError(v_target, v_pred).mean();
  });
}

async function train_step(batch_size = 64) {
    // Sample batch
    const x0 = make_data(batch_size)
    const x1 = make_batch(batch_size).arraySync();
    const t = tf.randomNormal([batch_size]).sigmoid().arraySync();
    // Train
    const l = await optimizer.minimize(() => flow_matching_loss(model, x0, x1, t), true);
    loss = l.dataSync()[0];
    loss_history.push(loss);
    batch_idx++;
    plot_loss();
}

async function train_loop() {
  if (!isTraining) return;
  await train_step();
  if (batch_idx % 10 === 0) {
    clear_plot();
    plot_points(X, TRUE_COLOR, 2);
    // Optionally, plot some reverse samples
  }
  setTimeout(train_loop, 0);
}

// --- Reverse Process Sampling ---
let sample_trajs = [];
const NUM_SAMPLES = 512;
const REVERSE_STEPS = 32;
const dt = 1.0 / REVERSE_STEPS;

async function generate_samples() {
    sample_trajs = [];
    let xt = make_batch(NUM_SAMPLES);
    let trajs = [xt.arraySync()];
    const timeSteps = tf.linspace(1, 0, REVERSE_STEPS).arraySync();
    for (let i = 0; i < timeSteps.length; i++) {
        const t = timeSteps[i];
        const t_col = tf.fill([NUM_SAMPLES, 1], t);
        const {newXt, arr} = tf.tidy(() => {
            const vt = model.predict([xt, t_col]);
            const newXt = tf.sub(xt, vt.mul(dt));
            const arr = newXt.arraySync();
            return {newXt, arr};
        });
        xt.dispose();
        xt = newXt;
        trajs.push(arr);
        t_col.dispose();
        // Allow UI to update every few steps
        if (i % 8 === 0) await new Promise(r => setTimeout(r, 0));
    }
    sample_trajs = trajs[0].map((_, i) => trajs.map(step => step[i]));
    stepSlider.value = 0;
    update_reverse_plot();
}

function update_reverse_plot() {
  clear_plot();
  plot_points(X, TRUE_COLOR, 2);
  const n_steps = parseInt(stepSlider.value);
  sliderLabel.textContent = `Reverse Step: ${n_steps}/${REVERSE_STEPS}`;
  if (sample_trajs.length === 0) return;
  for (let traj of sample_trajs) {
    plot_points([traj[Math.min(n_steps, traj.length-1)]], PRED_COLOR, 4);
  }
}

// --- UI Wiring ---
const toggleBtn = document.getElementById('toggleTrainBtn');

function updateToggleBtn() {
  const label = toggleBtn.querySelector('.btn-label');
  if (!label) return;
  if (isTraining) {
    label.textContent = 'Pause';
  } else if (batch_idx > 0) {
    label.textContent = 'Resume Training';
  } else {
    label.textContent = 'Start Training';
  }
}

toggleBtn.addEventListener('click', async (e) => {
    if (toggleBtn.disabled) return;
    if (!isTraining) {
        isTraining = true;
        // Disable and reset slider during training
        stepSlider.disabled = true;
        stepSlider.value = 0;
        sliderLabel.textContent = `Reverse Step: 0/${REVERSE_STEPS}`;
        clear_plot();
        plot_points(X, TRUE_COLOR, 2);
        train_loop();
    } else {
        isTraining = false;
        // Show loading spinner and disable button during sample generation
        toggleBtn.disabled = true;
        toggleBtn.classList.add('loading');
        await generate_samples();
        toggleBtn.classList.remove('loading');
        toggleBtn.disabled = false;
        stepSlider.disabled = false;
    }
    updateToggleBtn();
});

window.addEventListener('DOMContentLoaded', () => {
  const controlsDiv = document.getElementById('controls');
  // Create slider container
  const sliderContainer = document.createElement('div');
  sliderContainer.className = 'slider-container';
  // Create slider and label
  window.stepSlider = document.createElement('input');
  stepSlider.type = 'range';
  stepSlider.min = 0;
  stepSlider.max = REVERSE_STEPS;
  stepSlider.value = 0;
  stepSlider.id = 'stepSlider';
  window.sliderLabel = document.createElement('label');
  sliderLabel.htmlFor = 'stepSlider';
  sliderLabel.id = 'sliderLabel';
  sliderLabel.textContent = `Reverse Step: 0/${REVERSE_STEPS}`;
  sliderContainer.appendChild(sliderLabel);
  sliderContainer.appendChild(stepSlider);
  controlsDiv.appendChild(sliderContainer);
  stepSlider.oninput = update_reverse_plot;
  generate_samples(); // Compute samples on document load
  updateToggleBtn(); // Set initial button text
});

// --- Initial Plot ---
clear_plot();
plot_points(X, TRUE_COLOR, 2);
plot_loss();
