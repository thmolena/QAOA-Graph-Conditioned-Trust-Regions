const fadeElements = document.querySelectorAll(".fade-in");
const fadeObserver = new IntersectionObserver((entries) => {
  entries.forEach((entry) => {
    if (entry.isIntersecting) entry.target.classList.add("visible");
  });
}, { threshold: 0.15 });
fadeElements.forEach((element) => fadeObserver.observe(element));

const copyButton = document.getElementById("copyBibtex");
const bibtexCode = document.getElementById("bibtexCode");
copyButton.addEventListener("click", async () => {
  await navigator.clipboard.writeText(bibtexCode.innerText);
  copyButton.innerText = "Copied";
  setTimeout(() => { copyButton.innerText = "Copy"; }, 1600);
});
