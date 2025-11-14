/**
 * ===========================================================
 * Program: Card JS Template
 * Programmer/s: Cristina C. Villasor
 * Date Written: Nov. 13, 2025
 * Last Revised: Nov. 13, 2025
 *
 * Purpose: Handles generation of author and step cards and inserts them into HTML
 * 
 * Program Fits in the General System Design:
 * - Handles user interactions on the frontend
 * - Takes result from backend and displays it to user
 * 
 * Data Structure and Controls:
 * - Used arrays of objects to store author and step information
 * - Used ternary operators for conditional rendering
 * - Used for loops to iterate through the arrays
 * ===========================================================
 */
const authors = [
  {
    auth_name: "Ashley Faye J. Magcamit",
    initials: "AFM",
    role: "Computer Science Student",
    email: "mailto:ashleymagcamit@gmail.com",
    linkedin: "https://linkedin.com/in/your-profile",
    github: "https://github.com/fayethfuley",
  },
  {
    auth_name: "Catherine Joy R. Pailden",
    initials: "CJR",
    role: "Computer Science Student",
    email: "mailto:pailden.catherinejoy@gmail.com",
    linkedin: "https://linkedin.com/in/your-profile",
    github: "https://github.com/chickerinejoy",
  },
  {
    auth_name: "Alexandre C. Pornea",
    initials: "AP",
    role: "Computer Science Student",
    email: "mailto:alex041927@gmail.com",
    linkedin: "https://linkedin.com/in/your-profile",
    github: "https://github.com/alexandrepornea",
  },
  {
    auth_name: "Cristina C. Villasor",
    initials: "CV",
    role: "Computer Science Student",
    email: "mailto:cristinacvillasor960@gmail.com",
    linkedin: "https://www.linkedin.com/in/cristina-villasor-342448296",
    github: "https://github.com/ilhye",
  },
];

const steps = [
  {
    id: "01",
    title: "Upload Image",
    description:
      "Select and upload the image you want to analyze for adversarial patches",
  },
  {
    id: "02",
    title: "Process & Detect",
    description:
      "Our model processes the image using purification, super-resolution, and Faster R-CNN to detect potential inappropriate content.",
  },
  {
    id: "03",
    title: "View Results",
    description:
      "Review the analysis results with detailed insights on detected objects.",
  },
];

const auth_card = () => {
  let auth_card = "";

  for (let i = 0; i < authors.length; i++) {
    auth_card += `<div class="card sm:col-span-1">
      <div class="w-30 h-30 rounded-full bg-indigo-500 from-primary to-accent mx-auto mb-4 flex items-center justify-center text-white text-3xl font-bold">
        ${authors[i].initials}
      </div>
      <p class="font-semibold text-center mt-2">${authors[i].auth_name}</p>
      <p class="caption">${authors[i].role}</p>
      <div class="socials flex gap-2 justify-center">
          <a
              href="${authors[i].email}"
              class="bg-indigo-500 hover:bg-indigo-700 p-3 rounded-full text-white"
          >
              <svg
                xmlns="http://www.w3.org/2000/svg"
                width="24"
                height="24"
                viewBox="0 0 24 24"
                fill="none"
                stroke="currentColor"
                stroke-width="2"
                stroke-linecap="round"
                stroke-linejoin="round"
              >
                <path d="m22 7-8.991 5.727a2 2 0 0 1-2.009 0L2 7" />
                <rect x="2" y="4" width="20" height="16" rx="2" />
              </svg>
          </a>
          <a
            href="${authors[i].linkedin}"
            target="_blank"
            rel="noopener noreferrer"
            aria-label="Visit LinkedIn profile"
            class="bg-indigo-500 hover:bg-indigo-700 p-3 rounded-full text-white transition-colors duration-200"
          >
            <svg
              xmlns="http://www.w3.org/2000/svg"
              width="24"
              height="24"
              viewBox="0 0 24 24"
              fill="none"
              stroke="currentColor"
              stroke-width="2"
              stroke-linecap="round"
              stroke-linejoin="round"
              aria-hidden="true"
            >
              <path
                d="M16 8a6 6 0 0 1 6 6v7h-4v-7a2 2 0 0 0-2-2 2 2 0 0 0-2 2v7h-4v-7a6 6 0 0 1 6-6z"
              />
              <rect width="4" height="12" x="2" y="9" />
              <circle cx="4" cy="4" r="2" />
            </svg>
          </a>
          <a
            href="${authors[i].github}"
            target="_blank"
            rel="noopener noreferrer"
            aria-label="Visit GitHub profile"
            class="bg-indigo-500 hover:bg-indigo-700 p-3 rounded-full text-white transition-colors duration-200"
          >
            <svg
              xmlns="http://www.w3.org/2000/svg"
              width="24"
              height="24"
              viewBox="0 0 24 24"
              fill="none"
              stroke="currentColor"
              stroke-width="2"
              stroke-linecap="round"
              stroke-linejoin="round"
              class="lucide lucide-github-icon lucide-github"
            >
              <path
                d="M15 22v-4a4.8 4.8 0 0 0-1-3.5c3 0 6-2 6-5.5.08-1.25-.27-2.48-1-3.5.28-1.15.28-2.35 0-3.5 0 0-1 0-3 1.5-2.64-.5-5.36-.5-8 0C6 2 5 2 5 2c-.3 1.15-.3 2.35 0 3.5A5.403 5.403 0 0 0 4 9c0 3.5 3 5.5 6 5.5-.39.49-.68 1.05-.85 1.65-.17.6-.22 1.23-.15 1.85v4"
              />
              <path d="M9 18c-4.51 2-5-2-7-2" />
            </svg>
          </a>
      </div>
    </div>`;
  }
  return auth_card;
};

const step_card = () => {
  let step_card = "";

  for (let i = 0; i < steps.length; i++) {
    step_card += `<div class="relative">
      <!-- Connector line - hidden on mobile -->
      ${
        steps[i].id !== "03"
          ? '<div class="hidden md:block absolute top-8 left-[60%] w-[80%] h-0.5 bg-gradient-to-r from-indigo-500 to-purple-500 z-0"></div>'
          : ""
      }
      
      <!-- Step number badge - positioned to overlap the card -->
      <div
        class="absolute -top-6 left-1/2 -translate-x-1/2 w-12 h-12 bg-gradient-to-r from-indigo-500 to-purple-500 rounded-full flex items-center justify-center text-white font-bold z-10 shadow-lg"
      >
        ${steps[i].id}
      </div>

      <div class="card p-8 text-center pt-15 h-full">
        ${
          steps[i].id === "01"
            ? `<svg xmlns="http://www.w3.org/2000/svg" width="100" height="100" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round" class="lucide lucide-upload-icon lucide-upload m-auto text-indigo-500 mt-5">
                <path d="M12 3v12" />
                <path d="m17 8-5-5-5 5" />
                <path d="M21 15v4a2 2 0 0 1-2 2H5a2 2 0 0 1-2-2v-4" />
              </svg>`
            : steps[i].id === "02"
            ? `<svg xmlns="http://www.w3.org/2000/svg" width="100" height="100" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round" class="lucide lucide-scan-icon lucide-scan m-auto text-indigo-500 mt-5">
                <path d="M3 7V5a2 2 0 0 1 2-2h2" />
                <path d="M17 3h2a2 2 0 0 1 2 2v2" />
                <path d="M21 17v2a2 2 0 0 1-2 2h-2" />
                <path d="M7 21H5a2 2 0 0 1-2-2v-2" />
              </svg>`
            : steps[i].id === "03"
            ? `<svg xmlns="http://www.w3.org/2000/svg" width="100" height="100" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round" class="lucide lucide-circle-check-big-icon lucide-circle-check-big m-auto text-indigo-500 mt-5">
                <path d="M21.801 10A10 10 0 1 1 17 3.335" />
                <path d="m9 11 3 3L22 4" />
              </svg>`
            : ""
        }

        <h3 class="text-2xl font-semibold mb-3 mt-5">${steps[i].title}</h3>
        <p class="text-gray-500 font-light">${steps[i].description}</p>
      </div>
    </div>`;
  }

  return step_card;
};

console.log("tina");
document.getElementById("authors-info").innerHTML = auth_card();
document.getElementById("steps").innerHTML = step_card();