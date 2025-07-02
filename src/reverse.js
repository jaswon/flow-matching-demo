import { model } from './model.js';
import { make_batch } from './data.js';
import { clear_plot, plot_pred_points } from './plot.js';

export let sample_trajs = [];
export const NUM_SAMPLES = 512;
export const REVERSE_STEPS = 32;
export const dt = 1.0 / REVERSE_STEPS;

export async function generate_samples(stepSlider) {
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
        if (i % 8 === 0) await new Promise(r => setTimeout(r, 0));
    }
    sample_trajs = trajs[0].map((_, i) => trajs.map(step => step[i]));
    stepSlider.value = 0;
    update_reverse_plot(stepSlider, sliderLabel);
}

export function update_reverse_plot(stepSlider, sliderLabel) {
  clear_plot();
  const n_steps = parseInt(stepSlider.value);
  sliderLabel.textContent = `Reverse Step: ${n_steps}/${REVERSE_STEPS}`;
  if (sample_trajs.length === 0) return;
  for (let traj of sample_trajs) {
    plot_pred_points([traj[Math.min(n_steps, traj.length-1)]]);
  }
} 