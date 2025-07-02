
import * as tf from '@tensorflow/tfjs';

import { optimizer, model } from './model.js';

export function make_data(n_samples: number): [number,number][] {
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
        ] as [number,number]);
    }
    return X;
}

export function make_batch(batch_size: number) {
    // const W = 8;
    // return tf.randomUniform([batch_size, 2], -W/2, W/2);
    const STD = 1;
    return tf.randomNormal([batch_size, 2], 0, STD);
} 

export async function train_step(batch_size: number = 64) {
    const x0 = make_data(batch_size);
    const x1 = make_batch(batch_size).arraySync() as [number,number][];
    const t = tf.randomNormal([batch_size]).sigmoid().arraySync() as number[];
    return optimizer.minimize(() => tf.tidy(() => {
      const x0t = tf.tensor2d(x0);
      const x1t = tf.tensor2d(x1);
      const t_col = tf.tensor2d(t, [t.length, 1]);
      const xt = tf.add(tf.mul(t_col, x1t), tf.mul(tf.sub(1, t_col), x0t));
      const v_target = tf.sub(x1t, x0t);
      const v_pred = model.predict([xt, t_col]) as tf.Tensor;
      return tf.losses.meanSquaredError(v_target, v_pred).mean();
    }), true) as tf.Tensor;
}