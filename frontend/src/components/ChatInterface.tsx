import React, { useState, useEffect, useRef, useMemo } from 'react';
import {
    Box,
    TextField,
    Button,
    List,
    ListItem,
    Typography,
    Select,
    MenuItem,
    Paper,
    FormControl,
    InputLabel,
    IconButton,
    Divider,
    Stack
} from '@mui/material';
import SendIcon from '@mui/icons-material/Send';
import SettingsIcon from '@mui/icons-material/Settings';
import ReactMarkdown from 'react-markdown';
import remarkGfm from 'remark-gfm';
import { splitIntoSentences, stripMarkdown } from '../lib/sentence-utils';

interface Message {
    role: 'user' | 'assistant';
    content: string;
}

interface ChatInterfaceProps {
    ragApiUrl?: string;
    embedModel: string;
    chatSentences: any[];
    setChatSentences: (sentences: any[]) => void;
    currentChatId: number | null;
    activeSource: 'pdf' | 'chat';
    onJump: (id: number) => void;
}

const ChatInterface: React.FC<ChatInterfaceProps> = ({
    ragApiUrl = "http://localhost:8001",
    embedModel,
    chatSentences,
    setChatSentences,
    currentChatId,
    activeSource,
    onJump
}) => {
    const [messages, setMessages] = useState<Message[]>([]);
    const [input, setInput] = useState('');
    const [loading, setLoading] = useState(false);

    // Model selection
    const [llmModel, setLlmModel] = useState('');
    const [availableModels, setAvailableModels] = useState<string[]>([]);

    const messagesEndRef = useRef<null | HTMLDivElement>(null);
    const messageRefs = useRef<{ [key: number]: HTMLDivElement | null }>({});

    // Sync chatSentences with parent whenever messages change
    useEffect(() => {
        let globalId = 0;
        const result: { id: number; text: string; messageIndex: number }[] = [];
        messages.forEach((msg, mIdx) => {
            const stripped = stripMarkdown(msg.content);
            const sentences = splitIntoSentences(stripped);
            sentences.forEach((s) => {
                result.push({
                    id: globalId++,
                    text: s,
                    messageIndex: mIdx
                });
            });
        });
        setChatSentences(result);
    }, [messages, setChatSentences]);

    const activeMessageIndex = useMemo(() => {
        if (activeSource !== 'chat' || currentChatId === null) return null;
        return chatSentences[currentChatId]?.messageIndex;
    }, [currentChatId, chatSentences, activeSource]);

    useEffect(() => {
        if (activeMessageIndex !== null && messageRefs.current[activeMessageIndex]) {
            messageRefs.current[activeMessageIndex]?.scrollIntoView({
                behavior: 'smooth',
                block: 'center',
            });
        }
    }, [activeMessageIndex]);

    const scrollToBottom = () => {
        messagesEndRef.current?.scrollIntoView({ behavior: "smooth" });
    };

    useEffect(() => {
        scrollToBottom();
    }, [messages]);

    useEffect(() => {
        // Fetch models if possible
        fetch(`${ragApiUrl}/models`)
            .then(res => res.json())
            .then(data => {
                if (data.data) {
                    // Helper to extract ID
                    const ids = data.data.map((m: any) => m.id);
                    setAvailableModels(ids);
                }
            })
            .catch(err => {
                console.warn("Failed to fetch models", err);
                // Fallback
                setAvailableModels(['ai/qwen3:latest', 'ai/nomic-embed-text-v1.5:latest']);
            });
    }, [ragApiUrl]);

    const handleSend = async () => {
        if (!input.trim() || !llmModel || !embedModel) return;

        const userMsg: Message = { role: 'user', content: input };
        setMessages(prev => [...prev, userMsg]);
        setInput('');
        setLoading(true);

        try {
            const resp = await fetch(`${ragApiUrl}/chat`, {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json'
                },
                body: JSON.stringify({
                    question: userMsg.content,
                    history: messages,
                    llm_model: llmModel,
                    embedding_model: embedModel
                })
            });

            const data = await resp.json();
            const botMsg: Message = { role: 'assistant', content: data.answer };
            setMessages(prev => [...prev, botMsg]);
        } catch (err) {
            console.error(err);
            setMessages(prev => [...prev, { role: 'assistant', content: "Error: Failed to get response." }]);
        } finally {
            setLoading(false);
        }
    };

    return (
        <Paper elevation={0} sx={{ height: '100%', display: 'flex', flexDirection: 'column', p: 1, bgcolor: 'transparent' }}>
            <Box sx={{ mb: 1, pt: 1, display: 'flex', alignItems: 'center', justifyContent: 'space-between', gap: 2 }}>
                <Typography variant="subtitle1" sx={{ fontWeight: 'bold', whiteSpace: 'nowrap' }}>Chat with PDF</Typography>

                <Box sx={{ flexGrow: 1, maxWidth: '250px' }}>
                    <FormControl fullWidth size="small">
                        <InputLabel>LLM</InputLabel>
                        <Select
                            value={llmModel}
                            label="LLM"
                            onChange={(e) => setLlmModel(e.target.value)}
                        >
                            {availableModels.map(m => (
                                <MenuItem key={m} value={m}>{m}</MenuItem>
                            ))}
                        </Select>
                    </FormControl>
                </Box>
            </Box>

            <List sx={{ flexGrow: 1, overflow: 'auto', borderRadius: 1, mb: 1, p: 1 }}>
                {messages.map((msg, idx) => (
                    <ListItem key={idx} ref={el => messageRefs.current[idx] = el} alignItems="flex-start" sx={{
                        flexDirection: 'column',
                        alignItems: msg.role === 'user' ? 'flex-end' : 'flex-start',
                        px: 0,
                        py: 0.5
                    }}>
                        <Paper
                            sx={{
                                p: 1.5,
                                bgcolor: msg.role === 'user' ? 'primary.main' : 'grey.100',
                                color: msg.role === 'user' ? 'white' : 'text.primary',
                                maxWidth: '90%',
                                boxShadow: activeMessageIndex === idx ? '0 0 10px rgba(25, 118, 210, 0.5)' : 'none',
                                border: activeMessageIndex === idx ? '2px solid' : 'none',
                                borderColor: 'primary.main',
                                borderRadius: '12px',
                                transition: 'all 0.2s ease',
                                cursor: 'pointer'
                            }}
                            onDoubleClick={(e) => {
                                const firstSentence = chatSentences.find(s => s.messageIndex === idx);
                                if (firstSentence) onJump(firstSentence.id);
                                e.stopPropagation();
                            }}
                        >
                            <Typography variant="body2" component="div" sx={{
                                '& p': { m: 0, mb: 1 },
                                '& p:last-child': { mb: 0 },
                                '& ul, & ol': { pl: 2, m: 0, mb: 1 },
                                '& li': { mb: 0.5 },
                                '& h1, & h2, & h3': { fontSize: '1.1rem', fontWeight: 'bold', mb: 1, mt: 1 },
                                '& code': { bgcolor: msg.role === 'user' ? 'rgba(255,255,255,0.2)' : 'rgba(0,0,0,0.05)', px: 0.5, borderRadius: '4px', fontFamily: 'monospace' },
                                '& pre': { bgcolor: msg.role === 'user' ? 'rgba(255,255,255,0.2)' : 'rgba(0,0,0,0.05)', p: 1, borderRadius: '4px', overflowX: 'auto', mb: 1 }
                            }}>
                                <ReactMarkdown
                                    remarkPlugins={[remarkGfm]}
                                >
                                    {msg.content}
                                </ReactMarkdown>
                            </Typography>
                        </Paper>
                    </ListItem>
                ))}
                <div ref={messagesEndRef} />
            </List>

            <Box sx={{ display: 'flex', gap: 1 }}>
                <TextField
                    fullWidth
                    variant="outlined"
                    multiline
                    maxRows={10}
                    placeholder={!llmModel || !embedModel ? "Select LLM model first..." : "Ask a question..." + (input ? "\n(Shift+Enter for new line)" : "")}
                    value={input}
                    onChange={(e) => setInput(e.target.value)}
                    onKeyDown={(e) => {
                        if (e.key === 'Enter' && !e.shiftKey) {
                            e.preventDefault();
                            handleSend();
                        }
                    }}
                    disabled={loading || !llmModel || !embedModel}
                    sx={{
                        '& .MuiOutlinedInput-root': {
                            bgcolor: 'white',
                            '& fieldset': {
                                borderColor: 'primary.light',
                                borderWidth: '1px',
                            },
                            '&:hover fieldset': {
                                borderColor: 'primary.main',
                            },
                        },
                    }}
                />
                <IconButton color="primary" onClick={handleSend} disabled={loading || !llmModel || !embedModel}>
                    <SendIcon />
                </IconButton>
            </Box>
        </Paper>
    );
};

export default ChatInterface;
