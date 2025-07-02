
import * as tf from '@tensorflow/tfjs';

import { model } from './model.js';
import { make_batch } from './train.js';

export const NUM_SAMPLES = 512;
export const REVERSE_STEPS = 32;
export const dt = 1.0 / REVERSE_STEPS;

export async function generate_samples(): Promise<[number, number][][]> {
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
    return trajs[0].map((_, i) => trajs.map(step => step[i]))
}