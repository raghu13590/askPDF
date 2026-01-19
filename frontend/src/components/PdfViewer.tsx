import React, { useState, useEffect, useRef, useMemo } from 'react';
import { Document, Page, pdfjs } from 'react-pdf';
import { Box, Typography } from '@mui/material';
import 'react-pdf/dist/esm/Page/AnnotationLayer.css';
import 'react-pdf/dist/esm/Page/TextLayer.css';

// Set worker source
pdfjs.GlobalWorkerOptions.workerSrc = `//unpkg.com/pdfjs-dist@${pdfjs.version}/build/pdf.worker.min.mjs`;

type BBox = {
    page: number;
    x: number;
    y: number;
    width: number;
    height: number;
    page_height: number;
    page_width: number;
};

type Sentence = {
    id: number;
    text: string;
    bboxes: BBox[];
};

type Props = {
    pdfUrl: string;
    sentences: Sentence[];
    currentId: number | null;
    onJump: (id: number) => void;
    autoScroll: boolean;
    isResizing?: boolean;
};

const PdfViewer = React.memo(function PdfViewer({ pdfUrl, sentences, currentId, onJump, autoScroll, isResizing }: Props) {
    const [numPages, setNumPages] = useState<number | null>(null);
    const [pageWidth, setPageWidth] = useState<number>(600); // Actual width for PDF rendering
    const [scale, setScale] = useState<number>(1); // CSS scale for instant feedback
    const containerRef = useRef<HTMLDivElement>(null);
    const pdfContentRef = useRef<HTMLDivElement>(null); // For scaling
    const sentenceRefs = useRef<{ [key: number]: HTMLDivElement | null }>({});

    function onDocumentLoadSuccess({ numPages }: { numPages: number }) {
        setNumPages(numPages);
    }

    // Only update pageWidth when resizing ends, use CSS scale for visual feedback during resize
    useEffect(() => {
        if (!containerRef.current) return;

        if (!isResizing) {
            // On resize end, update pageWidth to match container
            const width = containerRef.current.offsetWidth - 40;
            setPageWidth(width);
            setScale(1);
        } else {
            // During resize, use CSS scale for visual feedback
            if (containerRef.current) {
                const width = containerRef.current.offsetWidth - 40;
                if (pageWidth > 0) {
                    setScale(width / pageWidth);
                }
            }
        }
        // eslint-disable-next-line react-hooks/exhaustive-deps
    }, [isResizing]);

    // Pre-calculate sentences by page for performance
    const sentencesByPage = useMemo(() => {
        const map: { [key: number]: (Sentence & { pageBBoxes: BBox[] })[] } = {};
        sentences.forEach(s => {
            s.bboxes.forEach(b => {
                if (!map[b.page]) map[b.page] = [];
                let entry = map[b.page].find(e => e.id === s.id);
                if (!entry) {
                    entry = { ...s, pageBBoxes: [] };
                    map[b.page].push(entry);
                }
                entry.pageBBoxes.push(b);
            });
        });
        return map;
    }, [sentences]);

    // Auto-scroll to active sentence
    useEffect(() => {
        if (autoScroll && currentId !== null && sentenceRefs.current[currentId]) {
            sentenceRefs.current[currentId]?.scrollIntoView({
                behavior: 'smooth',
                block: 'center',
            });
        }
    }, [currentId, autoScroll]);

    // Optimized overlay rendering using pre-calculated map
    const getPageOverlays = (pageNumber: number) => {
        const pageData = sentencesByPage[pageNumber] || [];
        return pageData.map((sentence) => (
            <div
                key={sentence.id}
                style={{
                    position: 'absolute',
                    top: 0,
                    left: 0,
                    width: '100%',
                    height: '100%',
                    pointerEvents: 'none',
                }}
            >
                {sentence.pageBBoxes.map((bbox, idx) => (
                    <div
                        key={idx}
                        ref={el => {
                            if (sentence.id === currentId && idx === 0) {
                                sentenceRefs.current[sentence.id] = el;
                            }
                        }}
                        style={{
                            position: 'absolute',
                            left: `${(bbox.x / bbox.page_width) * 100}%`,
                            top: `${((bbox.page_height - (bbox.y + bbox.height)) / bbox.page_height) * 100}%`,
                            width: `${(bbox.width / bbox.page_width) * 100}%`,
                            height: `${(bbox.height / bbox.page_height) * 100}%`,
                            backgroundColor: sentence.id === currentId ? 'rgba(255, 255, 0, 0.4)' : 'transparent',
                            cursor: 'pointer',
                            pointerEvents: 'auto',
                        }}
                        onDoubleClick={(e) => {
                            e.stopPropagation();
                            onJump(sentence.id);
                        }}
                        title={sentence.text}
                    />
                ))}
            </div>
        ));
    };

    return (
        <Box
            ref={containerRef}
            sx={{
                height: '100%',
                width: '100%',
                overflow: 'auto',
                display: 'flex',
                flexDirection: 'column',
                alignItems: 'stretch',
                bgcolor: 'transparent',
                p: 0,
                m: 0,
            }}
        >
            <div
                ref={pdfContentRef}
                style={{
                    transform: scale !== 1 ? `scale(${scale})` : undefined,
                    transformOrigin: 'top left',
                    transition: isResizing ? 'none' : 'transform 0.15s',
                    width: '100%',
                    margin: 0,
                    padding: 0,
                }}
            >
                <Document
                    file={pdfUrl}
                    onLoadSuccess={onDocumentLoadSuccess}
                    loading={<Typography>Loading PDF...</Typography>}
                    error={<Typography color="error">Failed to load PDF.</Typography>}
                >
                    {Array.from(new Array(numPages), (el, index) => (
                        <Box
                            key={`page_${index + 1}`}
                            sx={{
                                position: 'relative',
                                mb: 0,
                                width: '100%',
                                maxWidth: '100%',
                                '& canvas': {
                                    width: '100% !important',
                                    height: 'auto !important',
                                    display: 'block',
                                },
                                p: 0,
                                m: 0,
                                boxShadow: 'none',
                                border: 'none',
                                background: 'none',
                            }}
                        >
                            <Page
                                pageNumber={index + 1}
                                width={pageWidth}
                                renderAnnotationLayer={true}
                                renderTextLayer={true}
                            />
                            {/* Overlay Layer */}
                            <div
                                style={{
                                    position: 'absolute',
                                    top: 0,
                                    left: 0,
                                    width: '100%',
                                    height: '100%',
                                    zIndex: 10
                                }}
                            >
                                {getPageOverlays(index + 1)}
                            </div>
                        </Box>
                    ))}
                </Document>
            </div>
        </Box>
    );
});

export default PdfViewer;
