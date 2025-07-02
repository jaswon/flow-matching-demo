
export class RandomFourierFeatureEmbedding extends tf.layers.Layer {
    static className = 'RandomFourierFeatureEmbedding';
    constructor(dim, sigma = 1.0, config = {}) {
        super(config);
        this.dim = dim;
        this.sigma = sigma;
        this.built = false;
    }
    build(inputShape) {
        const halfDim = Math.floor(this.dim / 2);
        this.frequencies = this.addWeight(
            'frequencies',
            [inputShape[1], halfDim],
            'float32',
            tf.initializers.randomNormal({stddev: 1.0 / this.sigma}),
            null,
            false,
        );
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
            return tf.concat([tf.cos(shifted), tf.sin(shifted)], -1);
        });
    }
    getConfig() {
        const config = super.getConfig();
        config.dim = this.dim;
        config.sigma = this.sigma;
        return config;
    }
}
tf.serialization.registerClass(RandomFourierFeatureEmbedding);