import { execSync } from 'node:child_process';

// the commit the dashboard was built from, shown in the panel footers and the PDF
let commit = 'dev';
try { commit = execSync('git rev-parse --short HEAD', { stdio: ['ignore', 'pipe', 'ignore'] }).toString().trim(); } catch { /* not a git checkout */ }

export default { define: { __COMMIT__: JSON.stringify(commit) } };
