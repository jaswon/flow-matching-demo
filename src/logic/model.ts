// Model and custom layer

import * as tf from '@tensorflow/tfjs';
import { RandomFourierFeatureEmbedding } from './rff.js';

const optimizer = tf.train.adam(0.0001);

const model = (() => {
    const hDim = 512;
    const xt = tf.input({shape: [2]});
    const t = tf.input({shape: [1]});
    let tEmb = new RandomFourierFeatureEmbedding(64, 1.0).apply(t);
    tEmb = tf.layers.dense({units:256, activation:'swish'}).apply(tEmb);
    let x = tf.layers.dense({units:hDim}).apply(xt);
    for (let i = 0; i < 5; i++) {
        let r = tf.layers.layerNormalization().apply(x);
        r = tf.layers.multiply().apply([
            tf.layers.dense({units:hDim}).apply(r) as tf.Tensor,
            tf.layers.dense({units:hDim, activation:'swish'}).apply(r) as tf.Tensor,
        ]) // swiGLU
        r = tf.layers.dropout({rate:.3}).apply(r);
        r = tf.layers.dense({units:hDim}).apply(r);
        const alpha = tf.layers.dense({units:hDim}).apply(tEmb);
        r = tf.layers.multiply().apply([r as tf.Tensor, alpha as tf.Tensor]);
        x = tf.layers.add().apply([x as tf.Tensor,r as tf.Tensor]);
    }
    const output = tf.layers.dense({units: 2}).apply(x);
    
    // @ts-ignore
    return tf.model({inputs: [xt, t], outputs: output});
})();
model.summary();

export { optimizer, model }; 