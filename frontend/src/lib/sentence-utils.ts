/**
 * Splits a text into sentences.
 * A simple regex-based approach that handles common sentence endings.
 */
export function splitIntoSentences(text: string): string[] {
    if (!text) return [];
    // Split on punctuation followed by whitespace, keeping the punctuation
    // This handles sentences that don't end in punctuation by just keeping them as-is.
    const sentences = text
        .split(/(?<=[.!?])\s+/)
        .map(s => s.trim())
        .filter(s => s.length > 0);

    return sentences.length > 0 ? sentences : [text];
}

/**
 * Strips Markdown formatting from a string to prepare it for TTS.
 * Handles bold, italics, links, and simple code blocks.
 */
export function stripMarkdown(text: string): string {
    return text
        // Remove bold/italic
        .replace(/(\*\*|__)(.*?)\1/g, '$2')
        .replace(/(\*|_)(.*?)\1/g, '$2')
        // Remove links [text](url)
        .replace(/\[(.*?)\]\(.*?\)/g, '$1')
        // Remove inline code
        .replace(/`(.*?)`/g, '$1')
        // Remove code blocks
        .replace(/```[\s\S]*?```/g, '')
        // Remove headers
        .replace(/^#+\s+/gm, '')
        // Remove blockquotes
        .replace(/^\s*>\s+/gm, '')
        // Clean up multiple spaces
        .replace(/\s+/g, ' ')
        .trim();
}
