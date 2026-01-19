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
    fileHash: string | null;
    chatSentences: any[];
    setChatSentences: (sentences: any[]) => void;
    currentChatId: number | null;
    activeSource: 'pdf' | 'chat';
    onJump: (id: number) => void;
}

const ChatInterface: React.FC<ChatInterfaceProps> = ({
    ragApiUrl = "http://localhost:8001",
    embedModel,
    fileHash,
    chatSentences,
    setChatSentences,
    currentChatId,
    activeSource,
    onJump
}) => {
    const [messages, setMessages] = useState<Message[]>([]);
    const [input, setInput] = useState('');
    const [loading, setLoading] = useState(false);
    const [isModelWarming, setIsModelWarming] = useState(false);
    const [indexingStatus, setIndexingStatus] = useState<'checking' | 'indexing' | 'ready' | 'error'>('checking');
    const [collectionName, setCollectionName] = useState<string | null>(null);

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
                block: 'start',
            });
        }
    }, [activeMessageIndex, currentChatId]); // Added currentChatId to trigger on every sentence change/jump

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
                if (data.data && data.data.length > 0) {
                    // Helper to extract ID
                    const ids = data.data.map((m: any) => m.id);
                    setAvailableModels(ids);
                } else {
                    throw new Error("No models found from API");
                }
            })
            .catch(err => {
                console.error("Failed to fetch models or no models found", err);
                setAvailableModels([]);
                // Optionally, you could set an error state here to display in the UI
            });
    }, [ragApiUrl]);

    // Handle Collection Naming and Polling
    useEffect(() => {
        if (!embedModel || !fileHash) {
            setIndexingStatus('ready'); // or checking, but if no model/hash, nothing to index yet
            setCollectionName(null);
            return;
        }

        const baseModelName = embedModel.split(":")[0];
        const safeModelName = baseModelName.replace(/-/g, "_").replace(/\./g, "_").replace(/\//g, "_");
        const cName = `rag_${safeModelName}_${fileHash}`;
        setCollectionName(cName);
        setIndexingStatus('checking');

        let intervalId: ReturnType<typeof setInterval>;

        const checkStatus = async () => {
            try {
                const res = await fetch(`${ragApiUrl}/status?collection_name=${cName}`);
                const data = await res.json();
                if (data.status === 'ready') {
                    setIndexingStatus('ready');
                    return true; // stop polling
                } else {
                    setIndexingStatus('indexing');
                    return false;
                }
            } catch (e) {
                console.error("Status check failed", e);
                // Don't set error immediately, maybe retry?
                return false;
            }
        };

        const startPolling = async () => {
            const ready = await checkStatus();
            if (!ready) {
                intervalId = setInterval(async () => {
                    const isReady = await checkStatus();
                    if (isReady) clearInterval(intervalId);
                }, 2000);
            }
        };

        startPolling();

        return () => {
            if (intervalId) clearInterval(intervalId);
        };
    }, [embedModel, fileHash, ragApiUrl]);

    const handleSend = async () => {
        if (!input.trim() || !llmModel || !embedModel) return;

        const userMsg: Message = { role: 'user', content: input };
        setMessages(prev => [...prev, userMsg]);
        setInput('');
        setLoading(true);
        setIsModelWarming(false);

        try {
            // First check if models are ready to provide better UI feedback
            const checkModel = async (modelName: string) => {
                try {
                    const res = await fetch(`${ragApiUrl}/health/model?model=${encodeURIComponent(modelName)}`);
                    if (!res.ok) return true; // If endpoint is down or 404, don't show warming message
                    const data = await res.json();
                    return data.ready;
                } catch (e) {
                    return true; // Default to true if check fails, let backend handle it
                }
            };

            const [llmReady, embedReady] = await Promise.all([
                checkModel(llmModel),
                checkModel(embedModel)
            ]);

            if (!llmReady || !embedReady) {
                setIsModelWarming(true);
            }

            const resp = await fetch(`${ragApiUrl}/chat`, {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json'
                },
                body: JSON.stringify({
                    question: userMsg.content,
                    history: messages,
                    llm_model: llmModel,
                    embedding_model: embedModel,
                    collection_name: collectionName
                })
            });

            if (!resp.ok) {
                const errorData = await resp.json();
                throw new Error(errorData.detail || "Failed to get response");
            }

            const data = await resp.json();
            const botMsg: Message = { role: 'assistant', content: data.answer };
            setMessages(prev => [...prev, botMsg]);
        } catch (err: any) {
            console.error(err);
            setMessages(prev => [...prev, { role: 'assistant', content: `Error: ${err.message || "Failed to get response."}` }]);
        } finally {
            setLoading(false);
            setIsModelWarming(false);
        }
    };

    return (
        <Paper elevation={0} sx={{ height: '100%', display: 'flex', flexDirection: 'column', p: 1, bgcolor: 'transparent' }}>
            <Box sx={{ mb: 1, pt: 1, display: 'flex', alignItems: 'center', justifyContent: 'space-between', gap: 2 }}>
                <Typography variant="subtitle1" sx={{ fontWeight: 'bold', whiteSpace: 'nowrap' }}>
                    Chat with PDF
                    {indexingStatus === 'indexing' && (
                        <Typography component="span" variant="caption" sx={{ ml: 1, color: 'warning.main' }}>
                            (Indexing...)
                        </Typography>
                    )}
                </Typography>

                <Box sx={{ flexGrow: 1, maxWidth: '250px' }}>
                    <FormControl fullWidth size="small">
                        <InputLabel id="llm-label">Select LLM</InputLabel>
                        <Select
                            labelId="llm-label"
                            id="llm-select"
                            value={llmModel}
                            label="Select LLM"
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
                                boxShadow: activeMessageIndex === idx ? '0 0 10px rgba(255, 255, 0, 0.4)' : 'none',
                                border: 'none',
                                borderColor: 'transparent',
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

            <Box sx={{ display: 'flex', flexDirection: 'column', gap: 1 }}>
                {isModelWarming && (
                    <Typography variant="caption" sx={{ color: 'info.main', textAlign: 'center', fontStyle: 'italic' }}>
                        Bringing the AI model online, this may take a moment...
                    </Typography>
                )}
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
                        disabled={loading || !llmModel || !embedModel || indexingStatus !== 'ready'}
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
                    <IconButton color="primary" onClick={handleSend} disabled={loading || !llmModel || !embedModel || indexingStatus !== 'ready'}>
                        <SendIcon />
                    </IconButton>
                </Box>
            </Box>
        </Paper>
    );
};

export default ChatInterface;
