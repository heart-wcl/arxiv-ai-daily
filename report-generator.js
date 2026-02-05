/**
 * Report Generator for arXiv AI Papers
 * Generates a Chinese daily report with paper summaries
 */

const fs = require('fs');
const path = require('path');

// Configuration
const INPUT_FILE = path.join(__dirname, 'output', 'latest-papers.json');
const REPO_DIR = __dirname;
const OUTPUT_FILE = path.join(__dirname, 'output', 'daily-report.md');

/**
 * Simple Chinese translation for categories
 */
const categoryTranslations = {
    'cs.AI': '人工智能',
    'cs.LG': '机器学习',
    'cs.CL': '计算语言学',
    'cs.CV': '计算机视觉',
    'cs.NE': '神经计算'
};

/**
 * Generate one-line summary from paper summary
 */
function generateBriefSummary(summary) {
    // Clean up the summary
    let cleanSummary = summary
        .replace(/\s+/g, ' ')
        .replace(/\. /g, '. ')
        .trim();
    
    // Take first 100 characters for brief summary
    if (cleanSummary.length > 100) {
        cleanSummary = cleanSummary.substring(0, 100).replace(/[,，]$/, '') + '...';
    }
    
    return cleanSummary;
}

/**
 * Format authors list
 */
function formatAuthors(authors) {
    if (!authors || authors.length === 0) return '未知作者';
    
    if (authors.length <= 3) {
        return authors.join('、');
    }
    
    return `${authors.slice(0, 3).join('、')} 等`;
}

/**
 * Format date
 */
function formatDate(dateStr) {
    const date = new Date(dateStr);
    return date.toLocaleDateString('zh-CN', {
        year: 'numeric',
        month: 'long',
        day: 'numeric'
    });
}

/**
 * Generate the report
 */
async function main() {
    console.log('📝 Generating daily report...\n');
    
    // Read papers
    let papersData;
    try {
        const data = fs.readFileSync(INPUT_FILE, 'utf8');
        papersData = JSON.parse(data);
    } catch (e) {
        console.error('❌ Error reading papers file:', e.message);
        console.log('Please run arxiv-fetcher.js first.');
        process.exit(1);
    }
    
    const { papers, fetchDate } = papersData;
    
    // Generate report content
    const reportDate = new Date().toLocaleDateString('zh-CN', {
        year: 'numeric',
        month: 'long',
        day: 'numeric',
        weekday: 'long'
    });
    
    let report = `# 🤖 arXiv AI 每日论文精选\n\n`;
    report += `**日期**: ${reportDate}\n\n`;
    report += `> 本报告由 [arXiv AI Daily](https://arxiv.org/list/cs.AI/pastweek?show=1000) 自动生成\n\n`;
    report += `---\n\n`;
    report += `## 📊 今日精选 ${papers.length} 篇论文\n\n`;
    
    // Group papers by category
    const byCategory = {};
    for (const paper of papers) {
        const cat = paper.primaryCategory;
        if (!byCategory[cat]) byCategory[cat] = [];
        byCategory[cat].push(paper);
    }
    
    // Generate content by category
    for (const [cat, catPapers] of Object.entries(byCategory)) {
        const catName = categoryTranslations[cat] || cat;
        report += `### 📚 ${catName} (${cat})\n\n`;
        
        for (let i = 0; i < catPapers.length; i++) {
            const paper = catPapers[i];
            report += `#### ${i + 1}. ${paper.title}\n\n`;
            report += `**👥 作者**: ${formatAuthors(paper.authors)}\n\n`;
            report += `**📅 发布时间**: ${formatDate(paper.published)}\n\n`;
            report += `**📝 简介**: ${generateBriefSummary(paper.summary)}\n\n`;
            report += `**🔗 论文链接**: [arXiv](${paper.absUrl}) | [PDF](${paper.pdfUrl})\n\n`;
            report += `---\n\n`;
        }
    }
    
    // Add footer
    report += `\n---\n`;
    report += `**📁 GitHub 仓库**: [yaonie/arxiv-ai-daily](https://github.com/yaonie/arxiv-ai-daily)\n\n`;
    report += `---\n\n`;
    report += `*本报告每日自动更新*\n`;
    
    // Save report
    fs.writeFileSync(OUTPUT_FILE, report, 'utf8');
    
    console.log(`✅ Report generated: ${OUTPUT_FILE}\n`);
    console.log(`📄 Total papers: ${papers.length}\n`);
    
    // Also create a JSON version for Discord
    const discordJson = {
        date: reportDate,
        papers: papers.map((p, i) => ({
            number: i + 1,
            title: p.title,
            authors: formatAuthors(p.authors),
            summary: generateBriefSummary(p.summary),
            category: categoryTranslations[p.primaryCategory] || p.primaryCategory,
            arxivUrl: p.absUrl,
            pdfUrl: p.pdfUrl
        }))
    };
    
    const discordFile = path.join(__dirname, 'output', 'report-discord.json');
    fs.writeFileSync(discordFile, JSON.stringify(discordJson, null, 2), 'utf8');
    console.log(`✅ Discord JSON saved: ${discordFile}`);
    
    return report;
}

main().catch(console.error);
