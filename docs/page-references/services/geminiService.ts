import { GoogleGenAI, Type, Schema } from "@google/genai";
import { AgentAnalysis } from '../types';

const ai = new GoogleGenAI({ apiKey: process.env.API_KEY });

const analysisSchema: Schema = {
  type: Type.OBJECT,
  properties: {
    caption: {
      type: Type.STRING,
      description: "A concise, professional caption describing the scene for a dataset."
    },
    objects: {
      type: Type.ARRAY,
      items: {
        type: Type.OBJECT,
        properties: {
          label: { type: Type.STRING },
          confidence: { type: Type.NUMBER, description: "Confidence score between 0 and 1" }
        },
        required: ["label", "confidence"]
      }
    },
    tags: {
      type: Type.ARRAY,
      items: { type: Type.STRING },
      description: "Relevant keywords for search indexing."
    },
    reasoning: {
      type: Type.STRING,
      description: "Brief agentic reasoning on why these labels were chosen."
    },
    technicalDetails: {
      type: Type.OBJECT,
      properties: {
        lighting: { type: Type.STRING, description: "Assessment of lighting conditions (e.g., Natural, Low-light)" },
        composition: { type: Type.STRING, description: "Assessment of framing and angle" }
      },
      required: ["lighting", "composition"]
    }
  },
  required: ["caption", "objects", "tags", "reasoning", "technicalDetails"]
};

export const analyzeMedia = async (base64Data: string, mimeType: string): Promise<AgentAnalysis> => {
  try {
    const modelId = 'gemini-2.5-flash-image'; 

    const response = await ai.models.generateContent({
      model: modelId,
      contents: {
        parts: [
          {
            inlineData: {
              mimeType: mimeType,
              data: base64Data
            }
          },
          {
            text: "You are an advanced Computer Vision Agent. Analyze this image for a high-quality labeling dataset. Identify key objects, assess technical quality, and provide structured metadata."
          }
        ]
      },
      config: {
        responseMimeType: "application/json",
        responseSchema: analysisSchema,
        temperature: 0.4, // Lower temperature for more factual analysis
      }
    });

    const text = response.text;
    if (!text) throw new Error("No response from AI");

    return JSON.parse(text) as AgentAnalysis;

  } catch (error) {
    console.error("Gemini Analysis Failed:", error);
    throw error;
  }
};

// Helper to convert Blob to Base64
export const fileToBase64 = (file: File): Promise<string> => {
  return new Promise((resolve, reject) => {
    const reader = new FileReader();
    reader.readAsDataURL(file);
    reader.onload = () => {
      const result = reader.result as string;
      // Remove the data URL prefix (e.g., "data:image/jpeg;base64,")
      const base64 = result.split(',')[1];
      resolve(base64);
    };
    reader.onerror = error => reject(error);
  });
};

// Helper to extract a frame from a video file
export const extractVideoFrame = (videoFile: File): Promise<{ base64: string, previewUrl: string }> => {
    return new Promise((resolve, reject) => {
        const video = document.createElement('video');
        video.preload = 'metadata';
        video.onloadedmetadata = () => {
            video.currentTime = Math.min(1.0, video.duration / 2); // Capture at 1s or middle
        };
        video.onseeked = () => {
            const canvas = document.createElement('canvas');
            canvas.width = video.videoWidth;
            canvas.height = video.videoHeight;
            const ctx = canvas.getContext('2d');
            if (ctx) {
                ctx.drawImage(video, 0, 0, canvas.width, canvas.height);
                const dataUrl = canvas.toDataURL('image/jpeg', 0.8);
                const base64 = dataUrl.split(',')[1];
                resolve({ base64, previewUrl: dataUrl });
            } else {
                reject(new Error("Canvas context failed"));
            }
        };
        video.onerror = (e) => reject(e);
        video.src = URL.createObjectURL(videoFile);
    });
};
