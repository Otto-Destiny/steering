javascript:(async () => {
  /* Collect your own X bookmarks from the page you are already viewing.
     Runs entirely in your browser, in your session, only when you click it.
     Nothing is sent anywhere: it downloads a file you choose to upload. */
  if (!location.hostname.endsWith("x.com") && !location.hostname.endsWith("twitter.com")) {
    alert("Open https://x.com/i/bookmarks first, then click this bookmarklet.");
    return;
  }
  const found = new Map();
  const collect = () => {
    for (const anchor of document.querySelectorAll('a[href*="/status/"]')) {
      const match = anchor.getAttribute("href").match(/^\/([^/]+)\/status\/(\d+)/);
      if (match) found.set(match[2], `https://x.com/${match[1]}/status/${match[2]}`);
    }
  };
  let stagnant = 0;
  let previous = 0;
  /* X virtualizes the list, so links must be read before scrolling past them. */
  for (let i = 0; i < 400 && stagnant < 5; i += 1) {
    collect();
    stagnant = found.size === previous ? stagnant + 1 : 0;
    previous = found.size;
    window.scrollBy(0, window.innerHeight * 0.9);
    await new Promise((resolve) => setTimeout(resolve, 700));
  }
  collect();
  if (found.size === 0) {
    alert("No bookmarked posts found. Are you on https://x.com/i/bookmarks?");
    return;
  }
  const blob = new Blob([JSON.stringify([...found.values()], null, 2)], {
    type: "application/json",
  });
  const link = document.createElement("a");
  link.href = URL.createObjectURL(blob);
  link.download = "x-bookmarks.json";
  link.click();
  URL.revokeObjectURL(link.href);
  alert(`Saved ${found.size} bookmarked post(s) to x-bookmarks.json.`);
})();
