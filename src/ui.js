import { train_loop, isTraining, batch_idx, setIsTraining } from './train.js';
import { generate_samples, update_reverse_plot, REVERSE_STEPS } from './reverse.js';
import { clear_plot } from './plot.js';

export function setupUI() {
  const toggleBtn = document.getElementById('toggleTrainBtn');
  const stepSlider = document.getElementById('stepSlider');
  const sliderLabel = document.getElementById('sliderLabel');

  // Set slider max and label dynamically from REVERSE_STEPS
  stepSlider.max = REVERSE_STEPS;
  sliderLabel.textContent = `Reverse Step: 0/${REVERSE_STEPS}`;

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
      setIsTraining(true);
      stepSlider.disabled = true;
      stepSlider.value = 0;
      sliderLabel.textContent = `Reverse Step: 0/${REVERSE_STEPS}`;
      clear_plot();
      train_loop();
    } else {
      setIsTraining(false);
      toggleBtn.disabled = true;
      toggleBtn.classList.add('loading');
      await generate_samples(stepSlider, sliderLabel);
      toggleBtn.classList.remove('loading');
      toggleBtn.disabled = false;
      stepSlider.disabled = false;
    }
    updateToggleBtn();
  });

  stepSlider.oninput = () => update_reverse_plot(stepSlider, sliderLabel);
} 