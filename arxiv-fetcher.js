/**
 * arXiv AI Paper Fetcher
 * Fetches latest AI papers from arXiv
 */

const https = require('https');
const fs = require('fs');
const path = require('path');

// Configuration
const CATEGORIES = ['cs.AI', 'cs.LG', 'cs.CL', 'cs.CV', 'cs.NE'];
const MAX_RESULTS = 5;
const OUTPUT_DIR = path.join(__dirname, 'output');

// Create output directory if not exists
if (!fs.existsSync(OUTPUT_DIR)) {
    fs.mkdirSync(OUTPUT_DIR, { recursive: true });
}

/**
 * Fetch papers from arXiv API
 */
function fetchArxivPapers(category, maxResults = 5) {
    return new Promise((resolve, reject) => {
        const query = `cat:${category}`;
        const url = `https://export.arxiv.org/api/query?search_query=${encodeURIComponent(query)}&sortBy=submittedDate&sortOrder=descending&max_results=${maxResults}`;

        https.get(url, (res) => {
            let data = '';
            res.on('data', chunk => data += chunk);
            res.on('end', () => {
                try {
                    const papers = parseAtomFeed(data, category);
                    resolve(papers);
                } catch (e) {
                    reject(e);
                }
            });
        }).on('error', reject);
    });
}

/**
 * Parse Atom XML response from arXiv
 */
function parseAtomFeed(xml, category) {
    const papers = [];
    const entryRegex = /<entry>([\s\S]*?)<\/entry>/g;
    let match;

    while ((match = entryRegex.exec(xml)) !== null) {
        const entry = match[1];
        
        const id = extractTag(entry, 'id');
        const title = extractTag(entry, 'title').replace(/\n/g, ' ').trim();
        const summary = extractTag(entry, 'summary').replace(/\n/g, ' ').trim();
        const published = extractTag(entry, 'published');
        const authors = extractAuthors(entry);
        
        // Get primary category
        const categoryMatch = entry.match(/<arxiv:primary_category[^>]*term=["']([^"']+)["']/);
        const primaryCategory = categoryMatch ? categoryMatch[1] : category;

        papers.push({
            id,
            title,
            summary: summary.substring(0, 300) + (summary.length > 300 ? '...' : ''),
            authors,
            published,
            primaryCategory,
            pdfUrl: id.replace('http://arxiv.org/abs/', 'http://arxiv.org/pdf/') + '.pdf',
            absUrl: id
        });
    }

    return papers;
}

function extractTag(xml, tag) {
    const regex = new RegExp(`<${tag}>([\\s\\S]*?)</${tag}>`, 'i');
    const match = xml.match(regex);
    return match ? match[1].trim() : '';
}

function extractAuthors(xml) {
    const authors = [];
    const authorRegex = /<author>([\s\S]*?)<\/author>/g;
    let match;
    
    while ((match = authorRegex.exec(xml)) !== null) {
        const name = extractTag(match[1], 'name');
        if (name) authors.push(name);
    }
    
    return authors;
}

/**
 * Main function
 */
async function main() {
    const arg = process.argv[2] || '0';
    const paperIndex = parseInt(arg);
    
    console.log('🤖 Fetching latest AI papers from arXiv...\n');
    
    let allPapers = [];
    
    for (const category of CATEGORIES) {
        try {
            console.log(`📚 Fetching from ${category}...`);
            const papers = await fetchArxivPapers(category, MAX_RESULTS);
            allPapers = allPapers.concat(papers);
        } catch (e) {
            console.error(`❌ Error fetching ${category}:`, e.message);
        }
    }
    
    // Sort by date
    allPapers.sort((a, b) => new Date(b.published) - new Date(a.published));
    
    // Remove duplicates
    const uniquePapers = [];
    const seenIds = new Set();
    for (const paper of allPapers) {
        if (!seenIds.has(paper.id)) {
            seenIds.add(paper.id);
            uniquePapers.push(paper);
        }
    }
    
    // Select papers starting from index
    const selectedPapers = uniquePapers.slice(paperIndex * 10, (paperIndex + 1) * 10);
    
    // Save to file
    const outputPath = path.join(OUTPUT_DIR, 'latest-papers.json');
    fs.writeFileSync(outputPath, JSON.stringify({
        fetchDate: new Date().toISOString(),
        totalPapers: selectedPapers.length,
        startIndex: paperIndex,
        papers: selectedPapers
    }, null, 2));
    
    console.log(`\n✅ Saved ${selectedPapers.length} papers to ${outputPath}`);
    console.log(`📄 Papers ready from index ${paperIndex * 10} onwards\n`);
    
    return selectedPapers;
}

main().catch(console.error);
