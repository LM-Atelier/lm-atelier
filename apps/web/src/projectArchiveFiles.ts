/** What a chosen archive is, read from its first bytes rather than its name. */

// The first bytes of every passphrase-encrypted archive.
const ENCRYPTED_MAGIC = [0x4c, 0x4d, 0x41, 0x41, 0x52, 0x43, 0x48, 0x00];
// After the magic come the format version and the cipher suite, then the kind.
const KIND_OFFSET = ENCRYPTED_MAGIC.length + 2;

/** The kind byte an encrypted backup carries. */
export const ENCRYPTED_BACKUP_KIND = 3;

function head(file: File, length: number): Promise<Uint8Array> {
  return new Promise((resolve, reject) => {
    const reader = new FileReader();
    reader.onload = () => resolve(new Uint8Array(reader.result as ArrayBuffer));
    reader.onerror = () => reject(reader.error ?? new Error("The archive could not be read."));
    reader.readAsArrayBuffer(file.slice(0, length));
  });
}

function encrypted(bytes: Uint8Array): boolean {
  return bytes.length >= ENCRYPTED_MAGIC.length && ENCRYPTED_MAGIC.every((byte, index) => bytes[index] === byte);
}

export async function isEncryptedArchive(file: File): Promise<boolean> {
  return encrypted(await head(file, ENCRYPTED_MAGIC.length));
}

/** The kind an encrypted archive says it is, or null for a file that is not one. */
export async function encryptedArchiveKind(file: File): Promise<number | null> {
  const bytes = await head(file, KIND_OFFSET + 1);
  return encrypted(bytes) && bytes.length > KIND_OFFSET ? bytes[KIND_OFFSET] : null;
}
