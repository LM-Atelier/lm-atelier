using System;
using System.Diagnostics;
using System.IO;
using System.Windows.Forms;

internal static class LMAtelierLauncher
{
    [STAThread]
    private static int Main()
    {
        string installRoot = Environment.GetEnvironmentVariable("LM_ATELIER_INSTALL_ROOT");
        if (String.IsNullOrWhiteSpace(installRoot))
        {
            installRoot = Path.Combine(
                Environment.GetFolderPath(Environment.SpecialFolder.LocalApplicationData),
                "LMAtelier"
            );
        }

        string launcher = Path.Combine(installRoot, "start-lm-atelier.ps1");
        if (!File.Exists(launcher))
        {
            MessageBox.Show(
                "LM Atelier is not installed yet. Double-click Setup LM Atelier.exe first.",
                "LM Atelier",
                MessageBoxButtons.OK,
                MessageBoxIcon.Information
            );
            return 1;
        }

        try
        {
            ProcessStartInfo startInfo = new ProcessStartInfo();
            startInfo.FileName = PowerShellPath();
            startInfo.Arguments =
                "-NoProfile -ExecutionPolicy Bypass -File " + Quote(launcher);
            startInfo.WorkingDirectory = installRoot;
            startInfo.UseShellExecute = false;
            Process process = Process.Start(startInfo);
            if (process == null)
            {
                throw new InvalidOperationException("Windows did not start LM Atelier.");
            }
            process.WaitForExit();
            return process.ExitCode;
        }
        catch (Exception error)
        {
            MessageBox.Show(
                error.Message,
                "LM Atelier",
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
