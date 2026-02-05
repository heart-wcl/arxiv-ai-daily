/**
 * GitHub Push Script
 * Commits and pushes the generated report to GitHub
 */

const { execSync } = require('child_process');
const fs = require('fs');
const path = require('path');

// Configuration
const REPO_DIR = __dirname;
const GITHUB_REPO = 'https://github.com/yaonie/arxiv-ai-daily.git';

/**
 * Run git commands
 */
function gitCommand(cmd, cwd = REPO_DIR) {
    try {
        const result = execSync(cmd, { 
            cwd, 
            encoding: 'utf8',
            stdio: ['pipe', 'pipe', 'pipe']
        });
        return { success: true, output: result };
    } catch (e) {
        return { success: false, error: e.message, output: e.stdout, stderr: e.stderr };
    }
}

/**
 * Main function
 */
async function main() {
    console.log('🚀 Pushing to GitHub...\n');
    
    // Configure git
    console.log('⚙️  Configuring git...');
    gitCommand('git config user.name "arXiv AI Bot"', REPO_DIR);
    gitCommand('git config user.email "bot@arxiv-ai-daily"', REPO_DIR);
    
    // Check git status
    console.log('📊 Checking git status...');
    const status = gitCommand('git status', REPO_DIR);
    
    if (!status.success) {
        console.error('❌ Git not available:', status.error);
        return;
    }
    
    // Check if we need to initialize git (for new repo)
    const isNewRepo = !fs.existsSync(path.join(REPO_DIR, '.git'));
    
    if (isNewRepo) {
        console.log('📦 Initializing new git repository...');
        gitCommand('git init', REPO_DIR);
        gitCommand('git remote add origin https://github.com/yaonie/arxiv-ai-daily.git', REPO_DIR);
    }
    
    // Add files
    console.log('📁 Staging files...');
    gitCommand('git add -A', REPO_DIR);
    
    // Check if there are changes
    const diff = gitCommand('git diff --cached --stat', REPO_DIR);
    if (diff.output && diff.output.trim()) {
        console.log('Changes to commit:');
        console.log(diff.output);
    } else {
        console.log('✅ No changes to commit');
        return;
    }
    
    // Commit
    const date = new Date().toISOString().split('T')[0];
    const commitMessage = `📊 Update papers report - ${date}`;
    
    console.log('💾 Committing changes...');
    const commit = gitCommand(`git commit -m "${commitMessage}"`, REPO_DIR);
    
    if (!commit.success) {
        console.error('❌ Commit failed:', commit.error);
        return;
    }
    
    console.log('✅ Committed successfully');
    
    // Push
    console.log('🌐 Pushing to GitHub...');
    const push = gitCommand('git push origin main --force', REPO_DIR);
    
    if (!push.success) {
        console.error('❌ Push failed:', push.error);
        
        // Try with credential helper
        console.log('🔄 Trying with credential helper...');
        gitCommand('git config credential.helper store', REPO_DIR);
        const retryPush = gitCommand('git push origin main', REPO_DIR);
        
        if (!retryPush.success) {
            console.error('❌ Push still failed:', retryPush.error);
            return;
        }
    }
    
    console.log('\n✅ Successfully pushed to GitHub!');
    console.log(`📁 Repository: ${GITHUB_REPO}\n`);
}

main().catch(console.error);
