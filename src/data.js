// Data generation utilities

export function make_data(n_samples) {
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

export function make_batch(batch_size) {
    // const W = 8;
    // return tf.randomUniform([batch_size, 2], -W/2, W/2);
    const STD = 1;
    return tf.randomNormal([batch_size, 2], 0, STD);
} 