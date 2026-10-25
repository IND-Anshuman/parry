/**
 * Prefix root-relative Markdown links (`/docs/...`) with Astro's `base`, so
 * content can link with absolute paths and still work when the site is served
 * from a sub-path such as https://user.github.io/parry/.
 */
export function remarkBaseLinks(base) {
  const prefix = base.replace(/\/$/, '');
  return () => (tree) => {
    if (!prefix) return;
    const walk = (node) => {
      if ((node.type === 'link' || node.type === 'definition') && typeof node.url === 'string') {
        if (node.url.startsWith('/') && !node.url.startsWith('//') && !node.url.startsWith(`${prefix}/`)) {
          node.url = prefix + node.url;
        }
      }
      if (node.children) node.children.forEach(walk);
    };
    walk(tree);
  };
}
