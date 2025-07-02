import * as tf from '@tensorflow/tfjs';

export class RandomFourierFeatureEmbedding extends tf.layers.Layer {
    static className = 'RandomFourierFeatureEmbedding';
    dim: number;
    sigma: number;
    // @ts-ignore
    built: boolean;
    frequencies: any;
    phases: any;
    constructor(dim: number, sigma: number = 1.0, config: any = {}) {
        super(config);
        this.dim = dim;
        this.sigma = sigma;
        this.built = false;
    }
    build(inputShape: any) {
        const halfDim = Math.floor(this.dim / 2);
        this.frequencies = this.addWeight(
            'frequencies',
            [inputShape[1], halfDim],
            'float32',
            tf.initializers.randomNormal({stddev: 1.0 / this.sigma}),
            undefined,
            false,
        );
        this.phases = this.addWeight(
            'phases',
            [halfDim],
            'float32',
            tf.initializers.randomUniform({minval: 0, maxval: 2 * Math.PI}),
            undefined,
            false,
        );
        this.built = true;
    }
    computeOutputShape(inputShape: any) {
        return [inputShape[0], this.dim];
    }
    call(inputs: any) {
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