using System;
using System.Diagnostics;
using System.IO;
using System.Windows.Forms;

internal static class SetupLauncher
{
    [STAThread]
    private static int Main()
    {
        string bundleRoot = AppDomain.CurrentDomain.BaseDirectory;
        string installer = Path.Combine(bundleRoot, "packaging", "windows", "install.ps1");
        if (!File.Exists(installer))
        {
            MessageBox.Show(
                "Setup could not find the LM Atelier installation files. "
                    + "Keep this application inside the extracted release folder.",
                "LM Atelier Setup",
                MessageBoxButtons.OK,
                MessageBoxIcon.Error
            );
            return 1;
        }

        try
        {
            ProcessStartInfo startInfo = new ProcessStartInfo();
            startInfo.FileName = PowerShellPath();
            startInfo.Arguments =
                "-NoProfile -ExecutionPolicy Bypass -File " + Quote(installer);
            startInfo.WorkingDirectory = bundleRoot;
            startInfo.UseShellExecute = false;
            Process process = Process.Start(startInfo);
            if (process == null)
            {
                throw new InvalidOperationException("Windows did not start the installer.");
            }
            process.WaitForExit();
            if (process.ExitCode != 0)
            {
                MessageBox.Show(
                    "LM Atelier setup did not complete. Review the installer window for details.",
                    "LM Atelier Setup",
                    MessageBoxButtons.OK,
                    MessageBoxIcon.Error
                );
            }
            return process.ExitCode;
        }
        catch (Exception error)
        {
            MessageBox.Show(
                error.Message,
                "LM Atelier Setup",
                MessageBoxButtons.OK,
                MessageBoxIcon.Error
            );
            return 1;
        }
    }

    private static string PowerShellPath()
    {
        return Path.Combine(
            Environment.GetFolderPath(Environment.SpecialFolder.System),
            "WindowsPowerShell",
            "v1.0",
            "powershell.exe"
        );
    }

    private static string Quote(string value)
    {
        return "\"" + value.Replace("\"", "\\\"") + "\"";
    }
}
