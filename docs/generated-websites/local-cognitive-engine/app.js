const revealTargets = document.querySelectorAll('[data-reveal]');

const observer = new IntersectionObserver(
  entries => {
    for (const entry of entries) {
      if (entry.isIntersecting) {
        entry.target.classList.add('is-visible');
        observer.unobserve(entry.target);
      }
    }
  },
  { threshold: 0.16 }
);

for (const target of revealTargets) {
  observer.observe(target);
}
