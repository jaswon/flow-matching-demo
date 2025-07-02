import { optimizer, model } from './model.js';
import { make_data, make_batch } from './data.js';
import { plot_loss } from './plot_loss.js';

// --- Training Loop State ---
export let isTraining = false;
export let batch_idx = 0;
export const loss_history = [];

export const X = make_data(1024);

export function setIsTraining(val) { isTraining = val; }

export function flow_matching_loss(model, x0, x1, t) {
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

export async function train_step(batch_size = 64) {
    const x0 = make_data(batch_size);
    const x1 = make_batch(batch_size).arraySync();
    const t = tf.randomNormal([batch_size]).sigmoid().arraySync();
    const l = await optimizer.minimize(() => flow_matching_loss(model, x0, x1, t), true);
    loss_history.push(l.dataSync()[0]);
    batch_idx++;
}

export async function train_loop() {
  if (!isTraining) return;
  await train_step();
  plot_loss();
  setTimeout(train_loop, 0);
} 