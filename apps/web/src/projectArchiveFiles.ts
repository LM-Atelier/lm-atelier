/** What a chosen project archive is, read from its first bytes rather than its name. */

// The first bytes of every passphrase-encrypted archive.
const ENCRYPTED_MAGIC = [0x4c, 0x4d, 0x41, 0x41, 0x52, 0x43, 0x48, 0x00];

export function isEncryptedArchive(file: File): Promise<boolean> {
  return new Promise((resolve, reject) => {
    const reader = new FileReader();
    reader.onload = () => {
      const head = new Uint8Array(reader.result as ArrayBuffer);
      resolve(head.length === ENCRYPTED_MAGIC.length && ENCRYPTED_MAGIC.every((byte, index) => head[index] === byte));
    };
    reader.onerror = () => reject(reader.error ?? new Error("The archive could not be read."));
    reader.readAsArrayBuffer(file.slice(0, ENCRYPTED_MAGIC.length));
  });
}
