// Model and custom layer

import { RandomFourierFeatureEmbedding } from './rff.js';

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

export { optimizer, model }; 