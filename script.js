document.documentElement.classList.add("js");

const revealElements = Array.from(document.querySelectorAll(".fade-in"));
const revealAll = () => revealElements.forEach((element) => element.classList.add("visible"));
const reducedMotion = window.matchMedia
  ? window.matchMedia("(prefers-reduced-motion: reduce)").matches
  : false;

if ("IntersectionObserver" in window && !reducedMotion) {
  const observer = new IntersectionObserver(
    (entries, activeObserver) => {
      entries.forEach((entry) => {
        if (!entry.isIntersecting) return;
        entry.target.classList.add("visible");
        activeObserver.unobserve(entry.target);
      });
    },
    { rootMargin: "0px 0px -8%", threshold: 0.08 },
  );
  revealElements.forEach((element) => observer.observe(element));
} else {
  revealAll();
}

const copyButton = document.getElementById("copyBibtex");
const bibtexCode = document.getElementById("bibtexCode");
const copyStatus = document.getElementById("copyStatus");

if (copyButton && bibtexCode && copyStatus) {
  copyButton.addEventListener("click", async () => {
    try {
      const bibtex = bibtexCode.textContent ?? "";
      if (!navigator.clipboard || !window.isSecureContext) {
        throw new Error("Clipboard API unavailable");
      }
      await navigator.clipboard.writeText(bibtex);
      copyButton.textContent = "Copied";
      copyStatus.textContent = "BibTeX copied to the clipboard.";
    } catch (_error) {
      const selection = window.getSelection();
      const range = document.createRange();
      range.selectNodeContents(bibtexCode);
      selection?.removeAllRanges();
      selection?.addRange(range);
      copyButton.textContent = "Selected";
      copyStatus.textContent = "Automatic copy was unavailable. The BibTeX text is selected for manual copying.";
    }
    setTimeout(() => {
      copyButton.textContent = "Copy";
      copyStatus.textContent = "";
    }, 2000);
  });
}
