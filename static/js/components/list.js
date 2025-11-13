const nav_link = [
  {
    link: "#home",
    label: "Home",
  },
  {
    link: "#about",
    label: "About",
  },
  {
    link: "#how-to-use",
    label: "How to use",
  },
  {
    link: "#try-now",
    label: "Try now",
  },
  {
    link: "#authors",
    label: "Authors",
  },
];

const adv_def = [
  {
    title: "What is a harmful cloaked content?",
    description:
      "This is any content designed to deceive or manipulate viewers, often with malicious intent. Examples include images or videos that have been altered to mislead viewers or hide harmful content. This type of content is possible due to adversarial attacks. </br><span class='font-bold'>Adversarial attacks</span>are techniques used to fool machine learning models by introducing subtle perturbations to the input data. These attacks can cause models to make incorrect predictions or classifications, posing significant challenges for the deployment of AI systems in real-world applications.",
  },
  {
    title: "How does it work?",
    description:
      "Adversarial attacks work by subtly modifying the input data in a way that is imperceptible to humans but causes the model to make incorrect predictions. This can be achieved through various methods:",
  },
];

const adv_type = [
  {
    title: "Projected Gradient Descent (PGD):",
    description:
      'Iterative version of FGSM. It takes multiple, smaller steps, constantly checking that the adversarial example stays within a constrained "threat model"',
  },
  {
    title: "Fast Gradient Sign Method (FGSM):",
    description:
      "Find the direction that will most increase the loss (the model's error) and take one small step in that direction.",
  },
  {
    title: "Expectation Over Transformation (EoT):",
    description:
      "EoT assumes the adversary's image will be transformed (e.g., rotated, scaled, changed in lighting) before the model sees it.",
  },
  {
    title: "Carlini & Wagner (CW):",
    description:
      "It frames the problem as a sophisticated optimization: find the smallest possible perturbation that causes misclassification",
  },
  {
    title: "Backward Pass Differentiable Approximation (BPDA):",
    description:
      'Many early defenses (e.g., adding randomization, pre-processing with denoisers) were not truly robust; they just "broke" the gradient, making white-box attacks like PGD fail. BPDA is designed to circumvent these "obfuscated gradients."',
  },
];

const nav = () => {
  let nav = "";

  for (let k = 0; k < nav_link.length; k++) {
    nav += `
      <li>
        <a href="${nav_link[k].link}" class="hover:text-indigo-500 hover:border-b-2 hover:border-b-indigo-500">
          ${nav_link[k].label}
        </a>
      </li>
    `;
  }

  return nav;
};

const description = () => {
  let description = "";

  for (let j = 0; j < adv_def.length; j++) {
    description += `<li>
      <p class="text-lg/12 text-indigo-500">${adv_def[j].title}</p>
      <p class="text-base/7 text-gray-800">${adv_def[j].description}</p>
    </li>
    `;
  }
  return description;
};

const type = () => {
  let type = "";

  for (let i = 0; i < adv_type.length; i++) {
    type += `<li class="flex items-start">
      <span class="text-primary font-bold mr-3 text-indigo-500">${i + 1}.</span>
      <span><strong>${adv_type[i].title}</strong>${
      adv_type[i].description
    }</span>
    </li>`;
  }
  return type;
};

document.getElementById("nav-link").innerHTML = nav();
document.getElementById("adv-definition").innerHTML = description();
document.getElementById("adv-type").innerHTML = type();
