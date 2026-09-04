import { motion } from 'framer-motion';
import { FileImage, ScanText, UploadCloud } from 'lucide-react';
import { useRef, useState } from 'react';

interface Props {
  file?: File;
  busy: boolean;
  onFile: (file: File) => void;
}

export function Dropzone({ file, busy, onFile }: Props) {
  const input = useRef<HTMLInputElement>(null);
  const [dragging, setDragging] = useState(false);

  const accept = (candidate?: File) => {
    if (!candidate || !candidate.type.startsWith('image/')) return;
    onFile(candidate);
  };

  return (
    <motion.button
      type="button"
      className={`dropzone ${dragging ? 'is-dragging' : ''} ${file ? 'has-file' : ''}`}
      onClick={() => input.current?.click()}
      onDragOver={(event) => { event.preventDefault(); setDragging(true); }}
      onDragLeave={() => setDragging(false)}
      onDrop={(event) => {
        event.preventDefault();
        setDragging(false);
        accept(event.dataTransfer.files[0]);
      }}
      whileHover={{ y: -1 }}
      whileTap={{ scale: 0.995 }}
    >
      <input
        ref={input}
        hidden
        type="file"
        accept="image/png,image/jpeg,image/webp,image/bmp"
        onChange={(event) => accept(event.target.files?.[0])}
      />
      <div className="dropzone__content">
        <div className="dropzone__copy">
          <span className="dropzone__icon">
            {busy ? <ScanText size={20} /> : file ? <FileImage size={20} /> : <UploadCloud size={20} />}
          </span>
          <div>
            <strong>{file ? file.name : 'Drop an image to Glypho'}</strong>
            <span>{file ? formatBytes(file.size) : 'PNG, JPEG, WebP or BMP'}</span>
          </div>
        </div>
        <span className="dropzone__action">{file ? 'Replace' : 'Choose image'}</span>
      </div>
    </motion.button>
  );
}

function formatBytes(bytes: number): string {
  if (bytes < 1024 * 1024) return `${(bytes / 1024).toFixed(0)} KB`;
  return `${(bytes / 1024 / 1024).toFixed(1)} MB`;
}